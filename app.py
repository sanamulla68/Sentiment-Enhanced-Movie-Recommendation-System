import streamlit as st
import random
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Mood-Based Movie Recommender", layout="wide")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

@st.cache_data
def load_movies():
    df = pd.read_csv('data/latest_movies_with_genres.csv')
    df = df.dropna(subset=['overview'])
    df['genres'] = df['genres'].fillna('')
    return df[['title', 'overview', 'poster_path', 'genres']]

sentiment_model = load_model()
movies = load_movies()

# Mood to genre mapping
def map_mood_to_genres(mood):
    mood_map = {
        "fear": ["Thriller", "Horror"],
        "joy": ["Comedy", "Adventure"],
        "sad": ["Comedy", "Family"],
        "anger": ["Action", "Comedy"],
        "love": ["Romance", "Drama"],
        "nostalgia": ["Drama", "Family"],
        "anxiety": ["Fantasy", "Animation"]
    }
    for key, value in mood_map.items():
        if key in mood.lower():
            return value
    return []

def recommend_movies(user_input, sentiment_label, selected_genres, excluded_genres, shuffle=False):
    tfidf = TfidfVectorizer(stop_words='english')
    movie_overviews = movies['overview'].tolist()
    all_texts = [user_input] + movie_overviews
    tfidf_matrix = tfidf.fit_transform(all_texts)

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    movies['similarity'] = cosine_sim

    filtered_movies = movies.copy()

    if sentiment_label.upper() == "NEGATIVE":
        dark_keywords = r'\b(thriller|horror|murder|death|crime|killer|revenge)\b'
        uplift_keywords = r'\b(love|friendship|happy|family|joy|fun|dream|hope|adventure)\b'

        filtered_movies = filtered_movies[
            ~filtered_movies['overview'].str.contains(dark_keywords, case=False, regex=True, na=False)
        ]

        filtered_movies['boost'] = filtered_movies['overview'].str.contains(
            uplift_keywords, case=False, regex=True, na=False
        ).astype(int)

        filtered_movies['similarity'] += filtered_movies['boost'] * 0.2
        filtered_movies.drop(columns=['boost'], inplace=True)

    if not selected_genres:
        selected_genres = map_mood_to_genres(user_input)

    if selected_genres:
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(lambda g: any(sg in g for sg in selected_genres))]

    if excluded_genres:
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(lambda g: all(eg not in g for eg in excluded_genres))]

    top_candidates = filtered_movies.sort_values(by='similarity', ascending=False).head(30)
    if shuffle:
        top_candidates = top_candidates.sample(frac=1, random_state=random.randint(1, 10000))
    recommended = top_candidates.head(5)

    return recommended[['title', 'overview', 'poster_path', 'genres']]

def main():
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
    }
    [data-testid="stApp"] {
        background: url("https://i.pinimg.com/736x/16/c4/73/16c473ca08af3844513f707d83cdb99a.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    section.main > div {
        background-color: rgba(0, 0, 0, 0.75);
        padding: 2rem;
        border-radius: 12px;
    }
    .title {
        font-size: 3rem;
        font-weight: 900;
        color: #FFD700;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Trebuchet MS', sans-serif;
        text-shadow: 2px 2px 4px #000;
    }
    .movie-card {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 2rem;
        background: rgba(44, 47, 74, 0.85);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        animation: fadeInCard 0.8s ease-in-out;
    }
    @keyframes fadeInCard {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .movie-title {
        font-size: 1.5rem;
        color: #FFD700;
        margin-bottom: 0.25rem;
        font-weight: bold;
    }
    .movie-overview {
        font-size: 1rem;
        color: #e0e0e0;
        text-align: justify;
    }
    .stButton>button {
        background-color: #FFD700 !important;
        color: black !important;
        font-weight: bold;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ffcc00 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🎬 Emotion-Based Movie Recommender</div>', unsafe_allow_html=True)

    sample_inputs = [
        "I’m feeling really happy and want something adventurous!",
        "Feeling a bit down and need something uplifting.",
        "I'm anxious and want a cozy comfort movie.",
        "Feeling nostalgic — maybe something heartwarming?",
        "I’m sad and need a cheerful comedy.",
        "Feeling angry and want something calming or fun.",
        "I want a thriller with twists."
    ]

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "last_recommendations" not in st.session_state:
        st.session_state.last_recommendations = pd.DataFrame()
    if "sentiment_label" not in st.session_state:
        st.session_state.sentiment_label = ""

    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area("💬 Describe your mood or what you're feeling:", value=st.session_state.user_input, height=140)
    with col2:
        if st.button("🎲 Surprise Me"):
            st.session_state.user_input = random.choice(sample_inputs)
            st.rerun()

    genres_available = sorted(set(g for sublist in movies['genres'].dropna().str.split(', ') for g in sublist))
    selected_genres = st.multiselect("✅ Include genres:", genres_available)
    excluded_genres = st.multiselect("🚫 Exclude genres:", genres_available)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🎥 Recommend Movies"):
            user_input = st.session_state.user_input
            if not user_input.strip():
                st.warning("⚠️ Please enter a description of how you're feeling.")
                return

            with st.spinner("🧠 Analyzing sentiment..."):
                sentiment = sentiment_model(user_input)[0]

            st.session_state.sentiment_label = sentiment['label']
            emoji_map = {"POSITIVE": "😄", "NEGATIVE": "😢", "NEUTRAL": "😐"}
            st.success(f"🧠 Detected Sentiment: **{sentiment['label']} {emoji_map.get(sentiment['label'].upper(), '')}**")

            with st.spinner("🔍 Finding matching movies..."):
                st.session_state.last_recommendations = recommend_movies(user_input, sentiment['label'], selected_genres, excluded_genres, shuffle=False)

    with col2:
        if st.button("🔄 Refresh Recommendations") and st.session_state.last_recommendations is not None:
            st.session_state.last_recommendations = recommend_movies(
                st.session_state.user_input,
                st.session_state.sentiment_label,
                selected_genres,
                excluded_genres,
                shuffle=True
            )

    if not st.session_state.last_recommendations.empty:
        poster_base_url = "https://image.tmdb.org/t/p/w200"
        for _, row in st.session_state.last_recommendations.iterrows():
            poster_url = poster_base_url + row['poster_path'] if pd.notna(row['poster_path']) else None
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)

            col1, col2 = st.columns([1, 4])
            with col1:
                if poster_url:
                    st.image(poster_url, width=120)
                else:
                    st.markdown("🖼️ *No poster available*")
            with col2:
                st.markdown(f"<div class='movie-title'>🎬 {row['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#FFD700;font-weight:bold;'>🎭 Genres: {row['genres']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='movie-overview'>{row['overview']}</div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
