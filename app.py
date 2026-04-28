import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import implicit
from sklearn.metrics.pairwise import cosine_similarity
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SongMatch",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# LOAD MODEL & ARTIFACTS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS
# ─────────────────────────────────────────────
def get_popular_items(df, N=10, exclude=set()):
    popular = (df.groupby("id")["user_id"]
                 .nunique()
                 .reset_index()
                 .rename(columns={"user_id": "listener_count"})
                 .sort_values("listener_count", ascending=False))
    popular = popular[~popular["id"].isin(exclude)]
    max_count = popular["listener_count"].max()
    popular["score"] = popular["listener_count"] / max_count
    return [(row["id"], round(row["score"], 4)) for _, row in popular.head(N).iterrows()]

def build_user_profile(user_id, df_user, item_profiles, audio_features):
    user_songs = df_user[df_user["user_id"] == user_id][["id", "play_count"]]
    user_songs = user_songs[user_songs["id"].isin(item_profiles.index)]
    if len(user_songs) == 0:
        return None
    profiles = item_profiles.loc[user_songs["id"], audio_features]
    weights  = user_songs.set_index("id")["play_count"]
    return np.average(profiles, weights=weights, axis=0)

def recommend_cbf(user_id, df_user, item_profiles, audio_features, N=10):
    user_profile = build_user_profile(user_id, df_user, item_profiles, audio_features)
    if user_profile is None:
        return []
    seen_ids   = set(df_user[df_user["user_id"] == user_id]["id"])
    candidates = item_profiles[~item_profiles.index.isin(seen_ids)]
    similarities = cosine_similarity(
        user_profile.reshape(1, -1),
        candidates[audio_features].values
    )[0]
    top_idx    = np.argsort(similarities)[::-1][:N]
    top_ids    = candidates.index[top_idx]
    top_scores = similarities[top_idx]
    return list(zip(top_ids, top_scores))

def recommend_cold_start(user_id, df_user, item_profiles, audio_features, df_all, N=10):
    user_songs = df_user[df_user["user_id"] == user_id]
    n_songs    = len(user_songs)
    seen_ids   = set(user_songs["id"])
    if n_songs == 0:
        return get_popular_items(df_all, N=N, exclude=seen_ids)
    elif n_songs == 1:
        n_cbf = N // 2
        n_pop = N - n_cbf
        cbf_recs = recommend_cbf(user_id, df_user, item_profiles, audio_features, N=n_cbf)
        cbf_ids  = set(r[0] for r in cbf_recs)
        pop_recs = get_popular_items(df_all, N=n_pop, exclude=seen_ids | cbf_ids)
        return cbf_recs + pop_recs
    else:
        return recommend_cbf(user_id, df_user, item_profiles, audio_features, N=N)

def get_recommendation(user_id, N=10):
    n_interactions = user_n_interactions.get(user_id, 0)

    if user_id not in user_id_to_idx and n_interactions == 0:
        source = "popularity"
        recs   = get_popular_items(df_cold, N=N)
    elif n_interactions >= 5 and user_id in user_id_to_idx:
        source   = "ALS"
        user_idx = user_id_to_idx[user_id]
        rec_items, scores = als_model.recommend(
            userid=user_idx,
            user_items=train_matrix[user_idx],
            N=N,
            filter_already_liked_items=True
        )
        recs = [(item_map[item_idx], float(score))
                for item_idx, score in zip(rec_items, scores)]
    else:
        source = "Hybrid CBF"
        user_songs = df_cold[df_cold["user_id"] == user_id]
        recs = recommend_cold_start(user_id, user_songs, item_profiles, audio_features, df_cold, N=N)
        recs = [(song_id, float(score)) for song_id, score in recs]

    output = []
    for rank, (song_id, score) in enumerate(recs, 1):
        info = item_id_to_name.get(song_id, {})
        output.append({
            "rank"      : rank,
            "song_id"   : song_id,
            "trackname" : info.get("trackname", "Unknown"),
            "artistname": info.get("artistname", "Unknown"),
            "score"     : round(score, 4),
            "source"    : source
        })
    return output

def get_audio_profile(user_id):
    n_interactions = user_n_interactions.get(user_id, 0)
    if n_interactions >= 5 and user_id in user_id_to_idx:
        return None  # warm user — no profile needed
    profile = build_user_profile(user_id, df_cold, item_profiles, audio_features)
    return profile

def get_listen_history(user_id):
    n_interactions = user_n_interactions.get(user_id, 0)
    if n_interactions == 0:
        return []
    rows = df_cold[df_cold["user_id"] == user_id][["trackname", "artistname", "play_count"]]
    return rows.sort_values("play_count", ascending=False).head(5).to_dict("records")

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #121212;
    color: #ffffff;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { background: #000000 !important; }
section[data-testid="stSidebar"] > div { padding: 1.5rem 1rem; }

/* Sidebar */
.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 2rem;
}
.logo-circle {
    width: 32px; height: 32px; background: #1DB954;
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 16px;
}
.logo-title { font-family: 'Space Mono', monospace; font-size: 15px; font-weight: 700; color: #fff; }

.nav-item {
    padding: 9px 12px; border-radius: 6px;
    font-size: 13px; color: #b3b3b3; cursor: pointer;
    margin-bottom: 2px; transition: all 0.2s;
}
.nav-item.active { background: #282828; color: #fff; }
.nav-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #1DB954; margin-right: 8px; }

/* Main area */
.main-wrap { padding: 2rem 2.5rem; }
.greeting { font-size: 24px; font-weight: 600; margin-bottom: 1.5rem; }
.greeting .name { color: #1DB954; }

/* Search */
.search-wrap {
    background: #282828; border-radius: 10px;
    padding: 14px 18px; display: flex; align-items: center;
    gap: 10px; border: 1px solid #333; margin-bottom: 0.75rem;
}

/* Model badge */
.model-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 11px; font-weight: 500;
    margin-bottom: 1rem;
}
.badge-als { background: #1a3a2a; color: #1DB954; border: 1px solid #1DB954; }
.badge-cbf { background: #2a1a3a; color: #a78bfa; border: 1px solid #a78bfa; }
.badge-pop { background: #3a2a1a; color: #f59e0b; border: 1px solid #f59e0b; }

/* Song cards */
.songs-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 12px; margin-top: 0.5rem;
}
.song-card {
    background: #282828; border-radius: 10px;
    padding: 14px; border: 1px solid transparent;
    transition: all 0.2s; cursor: pointer;
}
.song-card:hover { border-color: #1DB954; background: #333; }
.song-emoji { font-size: 28px; margin-bottom: 8px; }
.song-title { font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 3px; line-height: 1.3; }
.song-artist { font-size: 12px; color: #b3b3b3; margin-bottom: 8px; }
.match-label { font-size: 10px; color: #1DB954; margin-bottom: 4px; font-weight: 500; }
.match-bar-bg { height: 3px; background: #333; border-radius: 2px; }
.match-bar-fill { height: 100%; border-radius: 2px; background: #1DB954; transition: width 0.6s ease; }

/* Audio profile */
.panel {
    background: #282828; border-radius: 10px;
    padding: 16px; height: 100%;
}
.panel-title { font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 14px; }

.feat-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.feat-label { font-size: 11px; color: #b3b3b3; width: 110px; flex-shrink: 0; }
.feat-track { flex: 1; height: 4px; background: #333; border-radius: 2px; }
.feat-fill { height: 100%; border-radius: 2px; background: #1DB954; }
.feat-val { font-size: 10px; color: #b3b3b3; width: 32px; text-align: right; }

/* History */
.history-item {
    background: #1e1e1e; border-radius: 8px; padding: 10px 12px;
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 6px; cursor: pointer;
    transition: background 0.2s;
}
.history-item:hover { background: #333; }
.history-icon { font-size: 18px; width: 32px; height: 32px; background: #1a3a2a; border-radius: 6px; display: flex; align-items: center; justify-content: center; }
.history-name { font-size: 12px; color: #fff; font-weight: 500; }
.history-sub { font-size: 11px; color: #b3b3b3; }

/* Now playing */
.now-playing {
    background: #282828; border-radius: 10px;
    padding: 14px 18px; display: flex; align-items: center;
    gap: 14px; border: 1px solid #333; margin-top: 1.5rem;
}
.np-thumb { font-size: 24px; }
.np-name { font-size: 13px; font-weight: 600; color: #fff; }
.np-artist { font-size: 11px; color: #b3b3b3; }
.progress-bar { height: 3px; background: #333; border-radius: 2px; margin-top: 6px; }
.progress-fill { height: 100%; background: #1DB954; border-radius: 2px; }

/* Section title */
.section-title { font-size: 16px; font-weight: 600; margin-bottom: 4px; }
.section-sub { font-size: 12px; color: #b3b3b3; margin-bottom: 12px; }

/* Stagger animation */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.song-card { animation: fadeInUp 0.4s ease both; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-circle">🎵</div>
        <span class="logo-title">SongMatch</span>
    </div>
    <div class="nav-item active"><span class="nav-dot"></span>Discover</div>
    <div class="nav-item">My Profile</div>
    <div class="nav-item">History</div>
    <div class="nav-item">Liked Songs</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#b3b3b3;margin-bottom:8px;">Recent User IDs</div>', unsafe_allow_html=True)

    # Show a few sample user IDs
    sample_warm = list(user_id_to_idx.keys())[:3]
    sample_cold = df_cold["user_id"].unique()[:2].tolist()

    for uid in sample_warm[:2]:
        short = uid[:16] + "..."
        if st.button(short, key=f"sidebar_{uid}", use_container_width=True):
            st.session_state.user_id_input = uid

    st.markdown('<div style="font-size:10px;color:#666;margin-top:8px;">Tip: paste any user_id from your dataset</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

# Greeting
st.markdown('<div class="greeting">Good evening, <span class="name">Indira</span> 👋</div>', unsafe_allow_html=True)

# Search / Input
col_input, col_btn = st.columns([4, 1])
with col_input:
    user_id = st.text_input(
        "",
        placeholder="Paste a user_id to get recommendations...",
        label_visibility="collapsed",
        key="user_id_input"
    )
with col_btn:
    find_btn = st.button("Find songs", use_container_width=True)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
EMOJIS = ["🎸", "🎹", "🎶", "🎵", "🎼", "🎺", "🎻", "🥁", "🎷", "🎤"]
COLORS = ["#1a3a2a", "#2a1a3a", "#3a2a1a", "#1a2a3a", "#3a1a2a", "#2a3a1a"]

if user_id or find_btn:
    uid = user_id.strip()
    if not uid:
        st.warning("Please enter a user_id first.")
    else:
        with st.spinner("Finding your songs..."):
            recs     = get_recommendation(uid, N=10)
            source   = recs[0]["source"] if recs else "unknown"
            n_inter  = user_n_interactions.get(uid, 0)
            history  = get_listen_history(uid)
            profile  = get_audio_profile(uid)

        # Model badge
        if source == "ALS":
            badge_class = "badge-als"
            badge_text  = f"🤖 Collaborative Filtering (ALS) · {n_inter} songs in history"
        elif source == "Hybrid CBF":
            badge_class = "badge-cbf"
            badge_text  = f"🎯 Hybrid Content-Based · {n_inter} song(s) in history"
        else:
            badge_class = "badge-pop"
            badge_text  = "⭐ Popularity-Based · New user"

        st.markdown(f'<div class="model-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)

        # Audio profile + History panel
        col_profile, col_history = st.columns([1.4, 1])

        with col_profile:
            if profile is not None:
                display_features = ["danceability", "energy", "acousticness",
                                    "valence", "speechiness", "instrumentalness"]
                feature_indices  = [audio_features.index(f) for f in display_features if f in audio_features]

                bars_html = ""
                for i, feat in zip(feature_indices, display_features):
                    val = float(profile[i])
                    bars_html += f"""
                    <div class="feat-row">
                        <span class="feat-label">{feat.capitalize()}</span>
                        <div class="feat-track">
                            <div class="feat-fill" style="width:{val*100:.0f}%"></div>
                        </div>
                        <span class="feat-val">{val:.2f}</span>
                    </div>"""

                st.markdown(f"""
                <div class="panel">
                    <div class="panel-title">🎛️ Your audio profile</div>
                    {bars_html}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="panel">
                    <div class="panel-title">🎛️ Audio profile</div>
                    <div style="font-size:12px;color:#b3b3b3;margin-top:8px;">
                        Warm user — recommendations powered by collaborative filtering
                        based on listening patterns of similar users.
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_history:
            if history:
                items_html = ""
                for h in history[:4]:
                    items_html += f"""
                    <div class="history-item">
                        <div class="history-icon">🎵</div>
                        <div>
                            <div class="history-name">{h['trackname'][:28]}</div>
                            <div class="history-sub">{h['artistname'][:22]} · {h['play_count']} plays</div>
                        </div>
                    </div>"""
                st.markdown(f"""
                <div class="panel">
                    <div class="panel-title">📻 Listen history</div>
                    {items_html}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="panel">
                    <div class="panel-title">📻 Listen history</div>
                    <div style="font-size:12px;color:#b3b3b3;margin-top:8px;">No history available for this user.</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Recommendations grid
        st.markdown('<div class="section-title">Recommended for you</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">— based on {source.lower()} · showing top {len(recs)}</div>', unsafe_allow_html=True)

        # Build grid HTML
        cards_html = '<div class="songs-grid">'
        for i, rec in enumerate(recs[:6]):
            score_pct = int(rec["score"] * 100) if rec["score"] <= 1.0 else min(int(rec["score"] / 2), 100)
            emoji     = EMOJIS[i % len(EMOJIS)]
            color     = COLORS[i % len(COLORS)]
            delay     = i * 0.06
            track     = rec["trackname"][:32] + ("…" if len(rec["trackname"]) > 32 else "")
            artist    = rec["artistname"][:24] + ("…" if len(rec["artistname"]) > 24 else "")

            cards_html += f"""
            <div class="song-card" style="animation-delay:{delay}s">
                <div style="font-size:28px;width:100%;aspect-ratio:1;background:{color};
                     border-radius:8px;display:flex;align-items:center;
                     justify-content:center;margin-bottom:8px;">{emoji}</div>
                <div class="song-title">{track}</div>
                <div class="song-artist">{artist}</div>
                <div class="match-label">{score_pct}% match</div>
                <div class="match-bar-bg">
                    <div class="match-bar-fill" style="width:{score_pct}%"></div>
                </div>
            </div>"""

        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        # Full list expander
        with st.expander("See all 10 recommendations"):
            for rec in recs:
                score_pct = int(rec["score"] * 100) if rec["score"] <= 1.0 else min(int(rec["score"] / 2), 100)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;padding:8px 0;
                     border-bottom:1px solid #282828;">
                    <span style="font-size:13px;color:#b3b3b3;min-width:20px;">#{rec['rank']}</span>
                    <span style="font-size:13px;color:#fff;flex:1;">{rec['trackname']}</span>
                    <span style="font-size:12px;color:#b3b3b3;">{rec['artistname']}</span>
                    <span style="font-size:11px;color:#1DB954;min-width:60px;text-align:right;">{score_pct}% match</span>
                </div>""", unsafe_allow_html=True)

        # Now playing (first rec)
        if recs:
            first = recs[0]
            score_pct = int(first["score"] * 100) if first["score"] <= 1.0 else min(int(first["score"] / 2), 100)
            st.markdown(f"""
            <div class="now-playing">
                <div class="np-thumb">🎵</div>
                <div style="flex:1">
                    <div class="np-name">{first['trackname'][:40]}</div>
                    <div class="np-artist">{first['artistname']}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width:{score_pct}%"></div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="width:20px;height:20px;border-radius:50%;background:#333;
                         display:flex;align-items:center;justify-content:center;font-size:9px;">⏮</div>
                    <div style="width:30px;height:30px;border-radius:50%;background:#1DB954;
                         display:flex;align-items:center;justify-content:center;font-size:11px;">▶</div>
                    <div style="width:20px;height:20px;border-radius:50%;background:#333;
                         display:flex;align-items:center;justify-content:center;font-size:9px;">⏭</div>
                </div>
            </div>""", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#b3b3b3;">
        <div style="font-size:48px;margin-bottom:1rem;">🎵</div>
        <div style="font-size:16px;font-weight:500;color:#fff;margin-bottom:8px;">
            Paste a user ID to get started
        </div>
        <div style="font-size:13px;">
            We'll find songs you'll love based on your listening history
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
