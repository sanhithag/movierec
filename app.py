import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. PAGE SETUP
st.set_page_config(page_title="FilmMatch AI", page_icon="🎬", layout="wide")

# 2. CUSTOM CSS (Enhanced for 9-grid stability)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .movie-card {
        background: #1e1e26;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #333;
        transition: transform 0.3s ease;
        margin-bottom: 20px;
        /* Removed fixed height to allow descriptions to be visible */
    }
    .movie-card:hover {
        transform: translateY(-5px);
        border-color: #ff4b4b;
    }
    .rating-badge {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. ENGINE
@st.cache_resource
def load_engine():
    df = pd.read_pickle("movies_lite.pkl")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, similarity

df, similarity = load_engine()

# 4. OMDb API FETCH
def fetch_details(title):
    api_key = st.secrets["OMDB_API_KEY"]
    try:
        # Request full plot (plot=full) to ensure description isn't cut off
        res = requests.get(f"http://www.omdbapi.com/?t={title}&plot=full&apikey={api_key}")
        return res.json()
    except:
        return None

# 5. UI
st.title("🎬 FilmMatch AI")
selected_movie = st.selectbox("Search for a movie you love:", df['title'].values, index=None)

if selected_movie:
    with st.spinner('Building your watchlist...'):
        idx = df[df['title'] == selected_movie].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        
        # We take the top 9 recommendations
        recs = distances[1:10] 
        
        st.write("---")
        
        # We loop through the recommendations in chunks of 3 to create rows
        for i in range(0, len(recs), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(recs):
                    movie_idx = recs[i+j][0]
                    title = df.iloc[movie_idx].title
                    details = fetch_details(title)
                    
                    with cols[j]:
                        poster = details.get('Poster', "") if details else ""
                        if poster == "N/A" or not poster:
                            poster = "https://via.placeholder.com/300x450?text=No+Poster"
                        
                        plot = details.get('Plot', "No description available.") if details else "..."
                        rating = details.get('imdbRating', 'N/A') if details else "N/A"

                        # Display Card
                        st.markdown(f"""
                            <div class="movie-card">
                                <img src="{poster}" style="width:100%; border-radius:8px;">
                                <h3 style="color:white; font-size:1.1rem; margin-top:10px;">{title}</h3>
                                <p class="rating-badge">⭐ IMDb: {rating}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Use an expander for the plot so it doesn't push the grid down
                        with st.expander("Read Plot Description"):
                            st.write(plot)
