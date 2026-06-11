import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="AI 영화 추천", layout="wide")

# ─────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────
@st.cache_data
def load_data():

    ratings = pd.read_csv(
        "data/raw/ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        "data/raw/movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movieId", "title", "genres"]
    )

    ratings = ratings.sort_values("timestamp")

    split = int(len(ratings) * 0.8)
    train_df = ratings.iloc[:split]
    test_df = ratings.iloc[split:]

    return ratings, train_df, test_df, movies


# ─────────────────────────────
# 2. 간단 추천 모델 (CBF 안정 버전)
# ─────────────────────────────
@st.cache_resource
def build_model(train_df, movies):

    movies["genre_text"] = movies["genres"].str.replace("|", " ")

    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies["genre_text"])

    features = tfidf_matrix.toarray()

    id_to_idx = {mid: i for i, mid in enumerate(movies["movieId"])}

    user_profiles = {}

    for uid, group in train_df.groupby("userId"):
        idxs = [id_to_idx[m] for m in group["movieId"] if m in id_to_idx]

        if len(idxs) == 0:
            continue

        weights = group["rating"].values[:len(idxs)]
        weights = weights / (np.sum(weights) + 1e-8)

        user_profiles[uid] = np.dot(weights, features[idxs])

    global_mean = train_df["rating"].mean()

    return features, id_to_idx, user_profiles, movies, global_mean


# ─────────────────────────────
# 3. 추천 함수
# ─────────────────────────────
def recommend(user_id, train_df, features, id_to_idx,
              user_profiles, movies, gm, n=10):

    seen = set(train_df[train_df["userId"] == user_id]["movieId"])

    if user_id not in user_profiles:
        return movies.sample(n)

    u_vec = user_profiles[user_id]

    scores = []

    for mid, idx in id_to_idx.items():
        if mid in seen:
            continue

        m_vec = features[idx]

        sim = np.dot(u_vec, m_vec) / (
            np.linalg.norm(u_vec) * np.linalg.norm(m_vec) + 1e-8
        )

        scores.append((mid, gm + sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    top = scores[:n]

    result = movies[movies["movieId"].isin([t[0] for t in top])].copy()
    result["score"] = [t[1] for t in top]

    return result.sort_values("score", ascending=False)


# ─────────────────────────────
# 4. MAIN
# ─────────────────────────────
def main():

    st.title("🎬 AI 영화 추천 시스템 (Stable)")

    ratings, train_df, test_df, movies = load_data()

    features, id_to_idx, user_profiles, movies, gm = build_model(train_df, movies)

    user_id = st.selectbox("User ID", train_df["userId"].unique())

    if st.button("추천 실행"):

        result = recommend(
            user_id,
            train_df,
            features,
            id_to_idx,
            user_profiles,
            movies,
            gm
        )

        st.subheader("추천 결과")

        for _, row in result.head(10).iterrows():
            st.write(f"🎬 {row['title']}")

if __name__ == "__main__":
    main()
