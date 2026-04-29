import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import implicit
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
import requests
from spotipy.oauth2 import SpotifyClientCredentials

st.set_page_config(page_title="SpotYourVibe", page_icon="🎵", layout="wide", initial_sidebar_state="collapsed")

@st.cache_resource
def init_spotify():
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=st.secrets["SPOTIFY_CLIENT_ID"],
        client_secret=st.secrets["SPOTIFY_CLIENT_SECRET"]
    ))

sp = init_spotify()

@st.cache_data(ttl=86400)
def get_album_art_url(trackname, artistname):
    try:
        # iTunes Search API (lebih stabil untuk cover art)
        query = f"{trackname} {artistname}"
        url = "https://itunes.apple.com/search"

        params = {
            "term": query,
            "media": "music",
            "entity": "song",
            "limit": 1
        }

        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()

            if data.get("resultCount", 0) > 0:
                art = data["results"][0].get("artworkUrl100")

                if art:
                    # Upgrade resolution
                    return art.replace("100x100", "600x600")

    except Exception as e:
        print("Album art error:", e)

    return None
    
@st.cache_resource
def load_model():
    als = implicit.cpu.als.AlternatingLeastSquares.load("recommendation_model/als_model.npz")
    with open("recommendation_model/artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    train_matrix = load_npz("recommendation_model/train_matrix.npz")
    df_cold = pd.read_csv("recommendation_model/df_cold.csv")
    user_interaction_count = pd.read_csv("recommendation_model/user_interaction_count.csv")
    return als, artifacts, train_matrix, df_cold, user_interaction_count

als_model, artifacts, train_matrix, df_cold, user_interaction_count = load_model()
user_map        = artifacts["user_map"]
item_map        = artifacts["item_map"]
user_id_to_idx  = artifacts["user_id_to_idx"]
item_profiles   = artifacts["item_profiles"]
audio_features  = artifacts["audio_features"]
item_id_to_name = artifacts["item_id_to_name"]
user_n_interactions = user_interaction_count.set_index("user_id")["n_interactions"].to_dict()

def get_popular_items(df, N=10, exclude=set()):
    popular = (df.groupby("id")["user_id"].nunique().reset_index()
                 .rename(columns={"user_id":"listener_count"})
                 .sort_values("listener_count", ascending=False))
    popular = popular[~popular["id"].isin(exclude)]
    popular["score"] = popular["listener_count"] / popular["listener_count"].max()
    return [(row["id"], round(row["score"],4)) for _,row in popular.head(N).iterrows()]

def build_user_profile(user_id, df_user, item_profiles, audio_features):
    user_songs = df_user[df_user["user_id"]==user_id][["id","play_count"]]
    user_songs = user_songs[user_songs["id"].isin(item_profiles.index)]
    if len(user_songs)==0: return None
    profiles = item_profiles.loc[user_songs["id"], audio_features]
    weights  = user_songs.set_index("id")["play_count"]
    return np.average(profiles, weights=weights, axis=0)

def recommend_cbf(user_id, df_user, item_profiles, audio_features, N=10):
    user_profile = build_user_profile(user_id, df_user, item_profiles, audio_features)
    if user_profile is None: return []
    seen_ids   = set(df_user[df_user["user_id"]==user_id]["id"])
    candidates = item_profiles[~item_profiles.index.isin(seen_ids)]
    sims = cosine_similarity(user_profile.reshape(1,-1), candidates[audio_features].values)[0]
    top  = np.argsort(sims)[::-1][:N]
    return list(zip(candidates.index[top], sims[top]))

def recommend_cold_start(user_id, df_user, item_profiles, audio_features, df_all, N=10):
    user_songs = df_user[df_user["user_id"]==user_id]
    n = len(user_songs); seen = set(user_songs["id"])
    if n==0: return get_popular_items(df_all, N=N, exclude=seen)
    elif n==1:
        cbf = recommend_cbf(user_id, df_user, item_profiles, audio_features, N=N//2)
        pop = get_popular_items(df_all, N=N-N//2, exclude=seen|set(r[0] for r in cbf))
        return cbf + pop
    else: return recommend_cbf(user_id, df_user, item_profiles, audio_features, N=N)

def get_recommendation(user_id, N=10):
    n = user_n_interactions.get(user_id, 0)
    if user_id not in user_id_to_idx and n==0:
        source, recs = "popularity", get_popular_items(df_cold, N=N)
    elif n>=5 and user_id in user_id_to_idx:
        source = "ALS"
        uid_idx = user_id_to_idx[user_id]
        ri, sc = als_model.recommend(userid=uid_idx, user_items=train_matrix[uid_idx], N=N, filter_already_liked_items=True)
        recs = [(item_map[i], float(s)) for i,s in zip(ri,sc)]
    else:
        source = "Hybrid CBF"
        us = df_cold[df_cold["user_id"]==user_id]
        recs = [(s,float(sc)) for s,sc in recommend_cold_start(user_id, us, item_profiles, audio_features, df_cold, N=N)]
    output = []
    for rank,(song_id,score) in enumerate(recs,1):
        info = item_id_to_name.get(song_id,{})
        output.append({"rank":rank,"song_id":song_id,
                        "trackname":info.get("trackname","Unknown"),
                        "artistname":info.get("artistname","Unknown"),
                        "score":round(score,4),"source":source})
    return output

def get_audio_profile(user_id):
    n = user_n_interactions.get(user_id, 0)
    if n>=5 and user_id in user_id_to_idx: return None
    return build_user_profile(user_id, df_cold, item_profiles, audio_features)

def compute_score_pct(recs):
    """
    Normalisasi score ke persentase menggunakan min-max normalization.
    Rekomendasi terbaik = 100%, yang lain proporsional di bawahnya.
    """
    scores = [r["score"] for r in recs]
    min_s  = min(scores)
    max_s  = max(scores)
    if max_s == min_s:
        return {r["rank"]: 100 for r in recs}
    return {
        r["rank"]: int((r["score"] - min_s) / (max_s - min_s) * 100)
        for r in recs
    }

# ── CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #222338;
    }

    .stApp {
        background: #e1e1e1;
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* HERO */
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #f3f3f3 100%);
        border: 1px solid rgba(34,35,56,0.1);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 0.05em;
        color: #0a66b7;
        margin-bottom: 0.3rem;
    }

    .hero-sub {
        font-size: 0.95rem;
        color: #555;
    }

    /* CARD */
    .section-card {
        background: #ffffff;
        border: 1px solid rgba(34,35,56,0.1);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1rem;
        letter-spacing: 0.15em;
        color: #0a66b7;
        margin-bottom: 1.2rem;
    }

    /* INPUT */
    label {
        color: #222338 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #ccc !important;
        border-radius: 10px !important;
        color: #222338 !important;
    }

    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #0a66b7 !important;
        box-shadow: 0 0 0 2px rgba(10,102,183,0.15) !important;
    }

    /* BUTTON */
    .stButton > button {
        width: 100%;
        background: #0a66b7;
        color: white;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #084c87;
        transform: translateY(-1px);
    }

    /* RESULT */
    .result-card {
        background: #ffffff;
        border: 1px solid rgba(34,35,56,0.1);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .clv-label {
        font-size: 0.8rem;
        color: #666;
    }

    .clv-customer {
        font-size: 0.9rem;
        color: #0a66b7;
        font-weight: 600;
    }

    .clv-value {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'DM Mono', monospace;
        color: #222338;
    }

    /* SEGMENTS */
    .segment-card {
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }

    .seg-name {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.2rem;
    }

    .seg-label {
        font-size: 0.8rem;
        color: #666;
    }

    .seg-desc {
        font-size: 0.8rem;
        color: #777;
    }

    .seg-platinum { background:#eef5fb; border:1px solid #0a66b7; }
    .seg-gold     { background:#f7f3e8; border:1px solid #c9a227; }
    .seg-silver   { background:#f0f2f5; border:1px solid #999; }
    .seg-bronze   { background:#f9eee8; border:1px solid #cd7f32; }

    /* PILLS */
    .info-pill {
        background: #f0f0f0;
        border: 1px solid #ccc;
        color: #333;
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.75rem;
    }

    hr {
        border-color: rgba(0,0,0,0.1);
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.25rem;">
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#fce68f;letter-spacing:-0.02em;">SpotYourVibe</div>
        <div style="font-size:11px;color:rgba(252,230,143,0.55);text-transform:uppercase;letter-spacing:0.09em;margin-top:4px;">Not sure where to start? Try these</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:rgba(252,230,143,0.65);margin-bottom:10px;">Sample User IDs</div>', unsafe_allow_html=True)
    for uid in list(user_id_to_idx.keys())[:3]:
        if st.button(uid[:18]+"...", key=f"sb_{uid}", use_container_width=True):
            st.session_state.user_id_input = uid
    with st.expander("About User IDs"):
        st.markdown("""
        **User IDs in this app come from the training dataset**
        
        • Click one of the sample IDs above  
        • Or enter another dataset User ID if available  
        
        *These may not match public Spotify usernames.*
        """)
    st.markdown("""
    <div style="margin-top:1.5rem;padding:14px 12px;background:rgba(252,230,143,0.08);border-radius:12px;border:1px solid rgba(252,230,143,0.18);">
        <div style="font-size:11px;color:rgba(252,230,143,0.75);line-height:1.7;">Click any ID above to auto-fill, or paste your own User ID.</div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────
st.markdown('<div class="syv-header"><div class="syv-logo">🎵</div><div><div class="syv-brand">SpotYourVibe</div><div class="syv-tagline">AI-Powered Music Recommendations</div></div></div>', unsafe_allow_html=True)
st.markdown('<div class="greeting">What are we listening to today? 🎧</div>', unsafe_allow_html=True)

c1, c2 = st.columns([4,1])
with c1:
    user_id = st.text_input("", placeholder="Paste a dataset User ID or try a sample user...", label_visibility="collapsed", key="user_id_input")
with c2:
    find_btn = st.button("Find Vibe ✦", use_container_width=True)

FALLBACK_BG    = ["#f8edf6","#eee8f8","#f8f2e8","#e8f0f8","#f8e8ee","#eaf8e8"]
FALLBACK_EMOJI = ["🎸","🎹","🎶","🎵","🎼","🎺"]

if user_id or find_btn:
    uid = user_id.strip()
    if not uid:
        st.warning("Please enter a User ID first.")
    else:
        with st.spinner("Matching your vibe..."):
            recs    = get_recommendation(uid, N=10)
            source  = recs[0]["source"] if recs else "unknown"
            n_inter = user_n_interactions.get(uid, 0)
            profile = get_audio_profile(uid)
            for rec in recs:
                clean_track = str(rec["trackname"]).split("(")[0].split("-")[0].strip()
                clean_artist = str(rec["artistname"]).split(",")[0].strip()
                rec["album_art"] = get_album_art_url(clean_track, clean_artist)

        # Hitung score_pct sekali untuk semua rekomendasi (min-max normalization)
        score_pct_map = compute_score_pct(recs)

        if source=="ALS":          bc,bt = "badge-als", f"✦ Collaborative Filtering (ALS) · {n_inter} songs in history"
        elif source=="Hybrid CBF": bc,bt = "badge-cbf", f"◈ Content-Based Hybrid · {n_inter} song(s) in history"
        else:                      bc,bt = "badge-pop", "◉ Popularity-Based · New listener"
        st.markdown(f'<div class="model-badge {bc}">{bt}</div>', unsafe_allow_html=True)

        # Audio profile
        if profile is not None:
            display_features = ["danceability","energy","acousticness","valence","speechiness","instrumentalness"]
            fi = [audio_features.index(f) for f in display_features if f in audio_features]
            bars = ""
            for i,feat in zip(fi, display_features):
                val = float(profile[i])
                bars += f'<div class="feat-row"><span class="feat-label">{feat.capitalize()}</span><div class="feat-track"><div class="feat-fill" style="width:{val*100:.0f}%"></div></div><span class="feat-val">{val:.2f}</span></div>'
            cp, _ = st.columns([1,1.8])
            with cp:
                st.markdown(f'<div class="panel"><div class="panel-title">Your Audio Profile</div>{bars}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Recommended for You</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Based on {source.lower()} · Top {len(recs)} picks</div>', unsafe_allow_html=True)

        # 3-column cards
        cols = st.columns(3, gap="medium")
        for i, rec in enumerate(recs[:6]):
            score_pct = score_pct_map[rec["rank"]]
            track  = rec["trackname"][:30]+("…" if len(rec["trackname"])>30 else "")
            artist = rec["artistname"][:22]+("…" if len(rec["artistname"])>22 else "")
            art    = rec.get("album_art")

            with cols[i%3]:
                with st.container(border=True):
                    if art:
                        st.image(art, use_container_width=True)
                    else:
                        bg    = FALLBACK_BG[i%len(FALLBACK_BG)]
                        emoji = FALLBACK_EMOJI[i%len(FALLBACK_EMOJI)]
                        st.markdown(f'<div style="width:100%;aspect-ratio:1;background:{bg};border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:44px;">{emoji}</div>', unsafe_allow_html=True)
                    st.markdown(f'''
                        <div class="song-title">{track}</div>
                        <div class="song-artist">{artist}</div>
                        <div class="match-label">{score_pct}% match</div>
                        <div class="match-bar-bg"><div class="match-bar-fill" style="width:{score_pct}%"></div></div>
                    ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("See all 10 recommendations"):
            for rec in recs:
                score_pct = score_pct_map[rec["rank"]]
                st.markdown(f'<div class="rec-row"><span class="rec-rank">#{rec["rank"]}</span><span class="rec-track">{rec["trackname"]}</span><span class="rec-artist">{rec["artistname"]}</span><span class="rec-score">{score_pct}%</span></div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <span class="empty-icon">🎵</span>
        <div class="empty-title">What's your vibe today?</div>
        <div class="empty-desc">Enter a User ID to discover songs that match your listening soul.<br>Open the sidebar to try a sample user.</div>
        <div class="empty-pill">← Open sidebar for sample IDs</div>
    </div>
    """, unsafe_allow_html=True)
