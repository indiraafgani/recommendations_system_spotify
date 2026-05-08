# 🎵 SpotYourVibe
### AI-Powered Personalized Music Recommendation System

> *From your listening history to your perfect next song.*

---

## 📌 Overview

**SpotYourVibe** is a hybrid music recommendation system that delivers personalized song suggestions based on a user's listening history and audio preferences. The system intelligently routes each user to the most appropriate model depending on how much listening data is available — ensuring relevant recommendations for everyone, from power users to first-time listeners.

This project was built as part of the **Purwadhika Data Science & Machine Learning Bootcamp**.

---

## 🎯 Problem Statement

With millions of songs available, finding music that truly matches your taste is overwhelming. Existing systems often fail for users who are new or have limited listening history. SpotYourVibe addresses this by combining multiple recommendation strategies into a single, unified system.

---

## 🧠 How It Works

SpotYourVibe uses a **3-tier routing system** based on a user's number of interactions:

| User Type | Condition | Model Used |
|---|---|---|
| 🔥 **Warm User** | ≥ 5 interactions | Collaborative Filtering (ALS) |
| 🌤️ **Cold User** | 1–4 interactions | Hybrid Content-Based Filtering + Popularity |
| 🆕 **New User** | 0 interactions | Popularity-Based |

### Model Details

**1. Collaborative Filtering — ALS (Alternating Least Squares)**
- Used for warm users with sufficient listening history
- Learns latent user and item embeddings from implicit feedback (play counts)
- Built with the `implicit` library
- ALS parameters: `factors=256`, `alpha=10`, `regularization=0.1`, `iterations=50`
- **Precision@10: 0.2106 | Recall@10: 0.0537 | NDCG@10: 0.2268**

**2. Hybrid Content-Based Filtering**
- Used for cold users with limited interactions
- Builds a user audio profile by weighting 9 audio features: `valence`, `energy`, `danceability`, `acousticness`, `tempo`, `loudness`, `instrumentalness`, `speechiness`, `liveness`
- Combines cosine similarity-based CBF with popularity-based fallback
- **Precision@10: 0.0026 | Recall@10: 0.0263 | NDCG@10: 0.0115**

**3. Popularity-Based**
- Used for brand new users with zero listening history
- Recommends the most widely listened-to tracks in the dataset

---

## 📊 Dataset

| Dataset | Description |
|---|---|
| `user_data.csv` | User listening history with play counts per track |
| `master_track_features.csv` | Track audio features (Spotify-style), genres, and metadata |

**Data preprocessing steps:**
- Artist name normalization for cross-dataset matching
- Play count aggregation per user-track pair
- User segmentation into warm (≥5 interactions) and cold (<5 interactions) groups
- Audio feature extraction and item profile construction

---

## 🗂️ Project Structure

```
SpotYourVibe/
│
├── Recommendation_System_-_Spotify.ipynb   # Full modeling notebook
├── app.py                                  # Streamlit web application
│
├── recommendation_model/
│   ├── als_model.npz                       # Trained ALS model
│   ├── artifacts.pkl                       # User/item mappings & metadata
│   ├── train_matrix.npz                    # Sparse interaction matrix
│   ├── df_cold.csv                         # Cold user interaction data
│   └── user_interaction_count.csv          # Interaction counts for routing
│
├── user_data.csv                           # Raw user listening history
└── master_track_features.csv              # Track audio features
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/SpotYourVibe.git
cd SpotYourVibe
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn implicit scipy spotipy requests rapidfuzz pyarrow
```

### 3. Set up Spotify API credentials
Create a `.streamlit/secrets.toml` file:
```toml
SPOTIFY_CLIENT_ID = "your_spotify_client_id"
SPOTIFY_CLIENT_SECRET = "your_spotify_client_secret"
```
> You can get these from the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🖥️ App Features

- 🔍 **User ID lookup** — enter any user ID from the dataset to get instant recommendations
- 🎨 **Album art display** — fetches cover art via iTunes Search API
- 📊 **Audio profile panel** — visualizes your listening fingerprint across 6 audio dimensions
- 🏷️ **Model badge** — shows which recommendation model was used and why
- 📋 **Top 10 list** — expandable full recommendation list with match scores
- 💡 **Sample user IDs** — sidebar shortcuts to try the system instantly

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

**Key libraries:** `implicit` (ALS), `rapidfuzz` (fuzzy artist matching), `spotipy` (Spotify API), `scipy.sparse` (sparse matrix operations)

---

## 👤 Author

**Indurasmi Dian M.**
**Khaerun Nisa'tri Safaati**
**Indira Faisa Afgani**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/indira-faisa-afgani-886548218)
**Mutiara Ayu Alzahra Ramadhani**

---

## 📄 License

This project is for educational purposes as part of the Purwadhika Digital Technology School Data Science & Machine Learning program.
