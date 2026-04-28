import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import implicit
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

st.set_page_config(page_title="Spot Your Vibe", page_icon="🎵", layout="wide", initial_sidebar_state="collapsed")

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
        result = sp.search(q=f"track:{trackname} artist:{artistname}", type="track", limit=1)
        items = result["tracks"]["items"]
        if items:
            images = items[0]["album"]["images"]
            if images:
                return images[1]["url"] if len(images) > 1 else images[0]["url"]
    except Exception:
        pass
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

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Force background on every Streamlit layer */
html, body { background-color: #fce68f !important; }
.stApp { background-color: #fce68f !important; }
[data-testid="stAppViewContainer"] { background-color: #fce68f !important; }
[data-testid="stHeader"] { background-color: #fce68f !important; }
[data-testid="stMain"] { background-color: #fce68f !important; }
[class*="css"] { font-family: 'DM Sans', sans-serif !important; }

:root {
    --purple:#6c2a5f; --purple-mid:#8b3a7a; --purple-lite:#b06ba0;
    --yellow:#fce68f; --yellow-dim:#f5d96a;
    --bg:#fce68f; --card:#fffef0; --card-border:#e8d460;
    --text-hi:#1c0a1a; --text-mid:#4a2444; --text-lo:#7a5a70;
    --shadow:rgba(108,42,95,0.10); --shadow-md:rgba(108,42,95,0.18);
}

#MainMenu, footer { visibility:hidden; }
.block-container { padding:3rem 2.5rem 3rem !important; max-width:100% !important; background:#fce68f !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background:linear-gradient(160deg,#6c2a5f 0%,#4a1a42 100%) !important; }
section[data-testid="stSidebar"] > div { padding:2rem 1.5rem !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span { color:#fce68f !important; }
section[data-testid="stSidebar"] hr { border-color:rgba(252,230,143,0.25) !important; }
section[data-testid="stSidebar"] .stButton>button {
    background:rgba(252,230,143,0.12) !important;
    border:1px solid rgba(252,230,143,0.28) !important;
    color:#fce68f !important; font-size:11.5px !important;
    border-radius:8px !important;
}
section[data-testid="stSidebar"] .stButton>button:hover { background:rgba(252,230,143,0.28) !important; }

/* Input */
.stTextInput>div>div>input {
    background:#fffef0 !important; border:2px solid #e8d460 !important;
    color:#1c0a1a !important; border-radius:14px !important;
    font-size:14px !important; padding:13px 18px !important;
}
.stTextInput>div>div>input:focus { border-color:#6c2a5f !important; }
.stTextInput>div>div>input::placeholder { color:#7a5a70 !important; }

/* Button */
.stButton>button {
    background:linear-gradient(135deg,#6c2a5f 0%,#8b3a7a 100%) !important;
    color:#fce68f !important; border:none !important;
    border-radius:14px !important; font-family:'Syne',sans-serif !important;
    font-weight:700 !important; font-size:13px !important;
    padding:13px 20px !important;
}
.stButton>button:hover { opacity:0.9 !important; }

/* Expander */
.streamlit-expanderHeader {
    background:#fffef0 !important; border:1.5px solid #e8d460 !important;
    border-radius:12px !important; color:#4a2444 !important;
    font-family:'Syne',sans-serif !important; font-weight:600 !important;
}

/* st.image — remove default padding/border */
[data-testid="stImage"] { border-radius:12px !important; overflow:hidden !important; margin-bottom:0 !important; }
[data-testid="stImage"] img { border-radius:12px !important; display:block !important; }

/* Card column styling */
.card-col { background:#fffef0; border:1.5px solid #e8d460; border-radius:18px; padding:14px; box-shadow:0 2px 10px rgba(108,42,95,0.10); }

/* Header */
.syv-header { display:flex; align-items:center; gap:16px; margin-bottom:2rem; padding-bottom:1.5rem; border-bottom:2px solid #e8d460; }
.syv-logo { width:48px; height:48px; background:linear-gradient(135deg,#6c2a5f,#8b3a7a); border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:24px; flex-shrink:0; }
.syv-brand { font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#6c2a5f; letter-spacing:-0.03em; line-height:1; }
.syv-tagline { font-size:11.5px; color:#7a5a70; margin-top:4px; letter-spacing:0.07em; text-transform:uppercase; }
.greeting { font-family:'Syne',sans-serif; font-size:21px; font-weight:700; color:#1c0a1a; margin-bottom:1.25rem; }

/* Badge */
.model-badge { display:inline-flex; align-items:center; gap:7px; padding:6px 16px; border-radius:99px; font-size:11.5px; font-weight:600; margin-bottom:1.5rem; }
.badge-als { background:rgba(108,42,95,0.10); color:#6c2a5f; border:1.5px solid rgba(108,42,95,0.35); }
.badge-cbf { background:rgba(245,217,106,0.25); color:#6a4a00; border:1.5px solid #f5d96a; }
.badge-pop { background:rgba(176,107,160,0.12); color:#b06ba0; border:1.5px solid #b06ba0; }

/* Panel */
.panel { background:#fffef0; border:1.5px solid #e8d460; border-radius:18px; padding:20px 22px; }
.panel-title { font-family:'Syne',sans-serif; font-size:12px; font-weight:700; color:#7a5a70; margin-bottom:18px; letter-spacing:0.08em; text-transform:uppercase; }
.feat-row { display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.feat-label { font-size:11.5px; color:#7a5a70; width:120px; flex-shrink:0; }
.feat-track { flex:1; height:5px; background:#e8c94a; border-radius:3px; overflow:hidden; }
.feat-fill { height:100%; border-radius:3px; background:linear-gradient(90deg,#6c2a5f,#c06898); }
.feat-val { font-size:10.5px; color:#7a5a70; width:34px; text-align:right; }

/* Section */
.section-title { font-family:'Syne',sans-serif; font-size:19px; font-weight:700; color:#1c0a1a; margin-bottom:5px; }
.section-sub { font-size:12px; color:#7a5a70; margin-bottom:16px; }

/* Song card text */
.song-title { font-family:'Syne',sans-serif; font-size:13px; font-weight:700; color:#1c0a1a; margin:10px 0 3px; line-height:1.35; }
.song-artist { font-size:11.5px; color:#7a5a70; margin-bottom:10px; }
.match-label { font-size:10px; font-weight:700; color:#6c2a5f; margin-bottom:5px; font-family:'Syne',sans-serif; letter-spacing:0.05em; text-transform:uppercase; }
.match-bar-bg { height:4px; background:#e8c94a; border-radius:2px; overflow:hidden; }
.match-bar-fill { height:100%; border-radius:2px; background:linear-gradient(90deg,#6c2a5f,#f5d96a); }

/* Expander list */
.rec-row { display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid #e8d460; }
.rec-rank { font-family:'Syne',sans-serif; font-size:12px; font-weight:700; color:#7a5a70; min-width:26px; }
.rec-track { font-size:13px; color:#1c0a1a; flex:1; font-weight:500; }
.rec-artist { font-size:12px; color:#7a5a70; }
.rec-score { font-size:11px; font-weight:700; color:#6c2a5f; min-width:46px; text-align:right; font-family:'Syne',sans-serif; }

/* Empty */
.empty-state { text-align:center; padding:6rem 2rem 4rem; }
.empty-icon { font-size:72px; margin-bottom:1.5rem; display:block; }
.empty-title { font-family:'Syne',sans-serif; font-size:26px; font-weight:800; color:#1c0a1a; margin-bottom:10px; }
.empty-desc { font-size:14px; color:#7a5a70; line-height:1.7; max-width:380px; margin:0 auto; }
.empty-pill { display:inline-block; background:rgba(108,42,95,0.10); color:#6c2a5f; border:1.5px solid rgba(108,42,95,0.25); border-radius:99px; padding:6px 18px; font-size:12px; font-weight:600; margin-top:20px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.25rem;">
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#fce68f;letter-spacing:-0.02em;">Spot Your Vibe</div>
        <div style="font-size:11px;color:rgba(252,230,143,0.55);text-transform:uppercase;letter-spacing:0.09em;margin-top:4px;">Try a sample</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:rgba(252,230,143,0.65);margin-bottom:10px;">Sample User IDs</div>', unsafe_allow_html=True)
    for uid in list(user_id_to_idx.keys())[:3]:
        if st.button(uid[:18]+"...", key=f"sb_{uid}", use_container_width=True):
            st.session_state.user_id_input = uid
    st.markdown("""
    <div style="margin-top:1.5rem;padding:14px 12px;background:rgba(252,230,143,0.08);border-radius:12px;border:1px solid rgba(252,230,143,0.18);">
        <div style="font-size:11px;color:rgba(252,230,143,0.75);line-height:1.7;">Click any ID above to auto-fill, or paste your own User ID.</div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────
st.markdown('<div class="syv-header"><div class="syv-logo">🎵</div><div><div class="syv-brand">Spot Your Vibe</div><div class="syv-tagline">AI-Powered Music Recommendations</div></div></div>', unsafe_allow_html=True)
st.markdown('<div class="greeting">What are we listening to today? 🎧</div>', unsafe_allow_html=True)

c1, c2 = st.columns([4,1])
with c1:
    user_id = st.text_input("", placeholder="Paste a User ID to discover your vibe...", label_visibility="collapsed", key="user_id_input")
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
                rec["album_art"] = get_album_art_url(rec["trackname"], rec["artistname"])

        # Badge
        if source=="ALS":       bc,bt = "badge-als", f"✦ Collaborative Filtering (ALS) · {n_inter} songs in history"
        elif source=="Hybrid CBF": bc,bt = "badge-cbf", f"◈ Content-Based Hybrid · {n_inter} song(s) in history"
        else:                    bc,bt = "badge-pop", "◉ Popularity-Based · New listener"
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

        # 3-column cards using Streamlit native columns
        cols = st.columns(3, gap="medium")
        for i, rec in enumerate(recs[:6]):
            score_pct = int(rec["score"]*100) if rec["score"]<=1.0 else min(int(rec["score"]/2),100)
            track  = rec["trackname"][:30]+("…" if len(rec["trackname"])>30 else "")
            artist = rec["artistname"][:22]+("…" if len(rec["artistname"])>22 else "")
            art    = rec.get("album_art")

            with cols[i%3]:
                # Card container
                with st.container():
                    st.markdown(f'<div style="background:#fffef0;border:1.5px solid #e8d460;border-radius:18px;padding:14px;box-shadow:0 2px 10px rgba(108,42,95,0.10);">', unsafe_allow_html=True)
                    if art:
                        st.image(art, use_container_width=True)
                    else:
                        bg = FALLBACK_BG[i%len(FALLBACK_BG)]
                        emoji = FALLBACK_EMOJI[i%len(FALLBACK_EMOJI)]
                        st.markdown(f'<div style="width:100%;aspect-ratio:1;background:{bg};border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:44px;">{emoji}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="song-title">{track}</div><div class="song-artist">{artist}</div><div class="match-label">{score_pct}% match</div><div class="match-bar-bg"><div class="match-bar-fill" style="width:{score_pct}%"></div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("See all 10 recommendations"):
            for rec in recs:
                score_pct = int(rec["score"]*100) if rec["score"]<=1.0 else min(int(rec["score"]/2),100)
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
