import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re

st.set_page_config(
    page_title="AI 영화 추천 시스템",
    page_icon="🎬",
    layout="wide"
)

# ── 데이터 로드 ────────────────────────────────────────
@st.cache_data
def load_models(train_df, movies):

    movie_stats = train_df.groupby('movieId').agg(
        avg_rating=('rating','mean'),
        rating_count=('rating','count')
    ).reset_index()

    movies_m = movies.merge(movie_stats, on='movieId', how='left')

    movies_m['avg_rating'] = movies_m['avg_rating'].fillna(
        movies_m['avg_rating'].median()
    )
    movies_m['rating_count'] = movies_m['rating_count'].fillna(0)

    movies_m['year'] = movies_m['year'].fillna(movies_m['year'].median())

    year_min = movies_m['year'].min()
    year_max = movies_m['year'].max()

    movies_m['year_norm'] = (movies_m['year'] - year_min) / (year_max - year_min + 1)

    movies_m['pop_norm'] = np.log1p(movies_m['rating_count'])
    movies_m['pop_norm'] = movies_m['pop_norm'] / movies_m['pop_norm'].max()

    tfidf = TfidfVectorizer()
    tfidf_mat = tfidf.fit_transform(movies_m['genre_text'])

    features = np.hstack([
        tfidf_mat.toarray() * 0.8,
        movies_m['year_norm'].values.reshape(-1,1) * 0.15,
        movies_m['pop_norm'].values.reshape(-1,1) * 0.05
    ])

    id_to_idx = {
        mid: idx for idx, mid in enumerate(movies_m['movieId'])
    }

    user_profiles = {}
    for uid, group in train_df.groupby('userId'):
        valid = group[group['movieId'].isin(id_to_idx)]
        if len(valid) == 0:
            continue

        idxs = [id_to_idx[m] for m in valid['movieId']]
        r = valid['rating'].values
        w = r / r.sum()

        user_profiles[uid] = np.dot(w, features[idxs])

    gm = movies_m['avg_rating'].mean()

    model_A = None  # 아직 SVD 없으면 임시

    return model_A, features, id_to_idx, user_profiles, movies_m, gm

model_A = None
# ── 추천 함수 ──────────────────────────────────────────
def recommend_svd(user_id, train_df, model_A, movies_m, n=10):
    seen = set(train_df[train_df['userId']==user_id]['movieId'])
    candidates = [m for m in movies_m['movieId'] if m not in seen]
    preds = [(m, model_A.predict(user_id, m).est) for m in candidates]
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:n]
    result = movies_m[movies_m['movieId'].isin([p[0] for p in top])].copy()
    score_map = {p[0]: p[1] for p in top}
    result['predicted_rating'] = result['movieId'].map(score_map)
    return result.sort_values('predicted_rating', ascending=False)

def recommend_cbf(user_id, train_df, features, id_to_idx,
                  user_profiles, movies_m, gm, n=10):
    seen = set(train_df[train_df['userId']==user_id]['movieId'])
    if user_id not in user_profiles:
        return movies_m.nlargest(n, 'avg_rating')
    u_vec = user_profiles[user_id]
    scores = []
    for mid, idx in id_to_idx.items():
        if mid in seen:
            continue
        m_vec  = features[idx]
        norm_u = np.linalg.norm(u_vec)
        norm_m = np.linalg.norm(m_vec)
        if norm_u == 0 or norm_m == 0:
            sim = 0
        else:
            sim = np.dot(u_vec, m_vec) / (norm_u * norm_m)
        movie_avg = movies_m[movies_m['movieId']==mid]['avg_rating'].values
        movie_avg = movie_avg[0] if len(movie_avg) > 0 else gm
        pred = np.clip(0.7*(gm + sim*2) + 0.3*movie_avg, 1, 5)
        scores.append((mid, pred))
    scores.sort(key=lambda x: x[1], reverse=True)
    top    = scores[:n]
    result = movies_m[movies_m['movieId'].isin([s[0] for s in top])].copy()
    score_map = {s[0]: s[1] for s in top}
    result['predicted_rating'] = result['movieId'].map(score_map)
    return result.sort_values('predicted_rating', ascending=False)

def recommend_hybrid(user_id, train_df, model_A, features, id_to_idx,
                     user_profiles, movies_m, gm, alpha=0.8, n=10):
    seen = set(train_df[train_df['userId']==user_id]['movieId'])
    candidates = [m for m in movies_m['movieId'] if m not in seen]

    if user_id not in user_profiles:
        u_vec = None
    else:
        u_vec = user_profiles[user_id]

    scores = []
    for mid in candidates:
        svd_est = model_A.predict(user_id, mid).est
        if u_vec is not None and mid in id_to_idx:
            m_vec  = features[id_to_idx[mid]]
            norm_u = np.linalg.norm(u_vec)
            norm_m = np.linalg.norm(m_vec)
            if norm_u > 0 and norm_m > 0:
                sim = np.dot(u_vec, m_vec) / (norm_u * norm_m)
            else:
                sim = 0
            movie_avg = movies_m[movies_m['movieId']==mid]['avg_rating'].values
            movie_avg = movie_avg[0] if len(movie_avg) > 0 else gm
            cbf_est = np.clip(0.7*(gm + sim*2) + 0.3*movie_avg, 1, 5)
        else:
            cbf_est = gm
        hybrid = np.clip(alpha*svd_est + (1-alpha)*cbf_est, 1, 5)
        scores.append((mid, hybrid))

    scores.sort(key=lambda x: x[1], reverse=True)
    top    = scores[:n]
    result = movies_m[movies_m['movieId'].isin([s[0] for s in top])].copy()
    score_map = {s[0]: s[1] for s in top}
    result['predicted_rating'] = result['movieId'].map(score_map)
    return result.sort_values('predicted_rating', ascending=False)

# ── 메인 앱 ────────────────────────────────────────────
def main():
    ratings, train_df, test_df, movies = load_data()

    with st.spinner("모델 로딩 중..."):
        model_A, features, id_to_idx, user_profiles, movies_m, gm = \
            load_models(train_df, movies)

    # 사이드바
    st.sidebar.title("🎬 영화 추천 시스템")
    page = st.sidebar.radio(
        "페이지 선택",
        ["홈", "추천받기", "모델 성능 비교"]
    )

    # ── 홈 페이지 ──
    if page == "홈":
        st.title("🎬 AI 기반 개인화 영화 추천 시스템")
        st.markdown("**시간 감쇠 SVD + CBF + Temporal NCF 하이브리드 추천 엔진**")

        st.divider()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("데이터셋", "MovieLens 1M")
        col2.metric("총 평점", "1,000,209개")
        col3.metric("유저 수", "6,040명")
        col4.metric("영화 수", "3,706개")

        st.divider()
        st.subheader("핵심 차별점")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("**시간 감쇠 SVD**\n\n오래된 평점의 영향력을 줄여 최근 취향을 더 잘 반영")
        with c2:
            st.info("**7개 모델 비교**\n\nA, B, C, A+B, A+C, B+C, A+B+C 전체 조합 Grid Search")
        with c3:
            st.info("**6개 지표 측정**\n\nRMSE, MAE, Precision, Recall, NDCG, ILD 동시 측정")

        st.divider()
        st.subheader("최종 모델 비교표")

        if os.path.exists('results/final_complete_v2.csv'):
            df = pd.read_csv('results/final_complete_v2.csv')
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("results/final_complete_v2.csv 파일이 없어요.")

    # ── 추천받기 페이지 ──
    elif page == "추천받기":
        st.title("🎯 영화 추천받기")

        user_ids = sorted(train_df['userId'].unique())
        user_id  = st.selectbox("유저 ID 선택", user_ids)

        model_choice = st.radio(
            "추천 모델 선택",
            ["A+C 하이브리드 (최고 성능)", "A: TemporalSVD", "B: CBF v2"],
            horizontal=True
        )

        if st.button("추천 받기", type="primary"):
            with st.spinner("추천 생성 중..."):
                if model_choice == "A+C 하이브리드 (최고 성능)":
                    result = recommend_hybrid(
                        user_id, train_df, model_A, features,
                        id_to_idx, user_profiles, movies_m, gm,
                        alpha=0.8
                    )
                elif model_choice == "A: TemporalSVD":
                    result = recommend_svd(
                        user_id, train_df, model_A, movies_m
                    )
                else:
                    result = recommend_cbf(
                        user_id, train_df, features, id_to_idx,
                        user_profiles, movies_m, gm
                    )

            st.subheader(f"유저 {user_id}에게 추천하는 영화 TOP 10")

            for i, (_, row) in enumerate(result.head(10).iterrows(), 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{i}. {row['title']}**")
                        st.caption(f"장르: {row['genres']}")
                    with col2:
                        st.metric("예상 평점",
                                  f"{row['predicted_rating']:.2f} / 5.0")
                st.divider()

            # 유저가 기존에 본 영화
            st.subheader("참고: 이 유저가 높게 평가한 영화")
            user_history = train_df[train_df['userId']==user_id].merge(
                movies_m[['movieId','title','genres']], on='movieId'
            ).sort_values('rating', ascending=False).head(5)
            st.dataframe(
                user_history[['title','genres','rating']],
                use_container_width=True
            )

    # ── 모델 성능 비교 페이지 ──
    elif page == "모델 성능 비교":
        st.title("📊 모델 성능 비교")

        if not os.path.exists('results/final_complete_v2.csv'):
            st.warning("results/final_complete_v2.csv 파일이 없어요.")
            return

        df = pd.read_csv('results/final_complete_v2.csv')

        # RMSE 차트
        st.subheader("RMSE 비교 (낮을수록 좋음)")
        fig1 = px.bar(
            df.sort_values('RMSE'),
            x='모델', y='RMSE',
            color='RMSE',
            color_continuous_scale='RdYlGn_r',
            text='RMSE'
        )
        fig1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)

        # Precision@10 + NDCG@10
        st.subheader("Precision@10 vs NDCG@10")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='Precision@10',
            x=df['모델'], y=df['Precision@10'],
            marker_color='#4C78A8'
        ))
        fig2.add_trace(go.Bar(
            name='NDCG@10',
            x=df['모델'], y=df['NDCG@10'],
            marker_color='#72B7B2'
        ))
        fig2.update_layout(barmode='group', height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # 정확도 vs 다양성
        st.subheader("정확도(RMSE) vs 다양성(ILD) 트레이드오프")
        fig3 = px.scatter(
            df, x='RMSE', y='ILD',
            text='모델', size_max=15,
            color='모델'
        )
        fig3.update_traces(textposition='top center')
        fig3.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        # 전체 지표 테이블
        st.subheader("전체 지표 상세")
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()