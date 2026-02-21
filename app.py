import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- PAGE CONFIG ---
st.set_page_config(page_title="Movierecs cuz im lazy ash", page_icon="🍿", layout="wide")

# --- SECURE API FETCHING ---
def get_poster(imdb_id):
    # This pulls the key from your Streamlit Dashboard Secrets
    try:
        api_key = st.secrets["TMDB_API_KEY"]
        url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
        data = requests.get(url).json()
        
        if data.get('movie_results'):
            path = data['movie_results'][0]['poster_path']
        elif data.get('tv_results'):
            path = data['tv_results'][0]['poster_path']
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster+Found"
            
        return f"https://image.tmdb.org/t/p/w500{path}"
    except Exception:
        return "https://via.placeholder.com/500x750?text=Logo+Missing"

# --- DATA ENGINE ---
@st.cache_data
def load_and_prep_data():
    # Load and keep only necessary columns to save RAM
    df = pd.read_csv('movies.csv', low_memory=False)
    
    # Filter: Modern movies only (>2000) and popular enough to be "real"
    df = df[df['startYear'] > 2000].copy()
    df['genres'] = df['genres'].fillna('')
    
    # Reset index so it matches the TF-IDF matrix exactly
    df = df.reset_index(drop=True)
    return df

@st.cache_resource 
def compute_tfidf(genres_series):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(genres_series)
    return matrix

# --- APP LAYOUT ---
try:
    df = load_and_prep_data()
    tfidf_matrix = compute_tfidf(df['genres'])

    st.title("🍿 Movierecs cuz im lazy ash")
    st.markdown("---")

    # Search functionality
    selected_title = st.selectbox("Search for a Movie/Show you liked:", df['primaryTitle'].values)

    if st.button('Recommend'):
        # Find index of selected movie
        idx = df[df['primaryTitle'] == selected_title].index[0]
        
        # Memory-efficient similarity (calculates only for the selected movie)
        cosine_sim_single = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Get top 6 matches (excluding the movie itself)
        sim_scores = sorted(list(enumerate(cosine_sim_single)), key=lambda x: x[1], reverse=True)[1:7]
        
        st.subheader(f"Because you liked '{selected_title}':")
        
        # Display in a clean 3-column grid
        cols = st.columns(3)
        for i, (m_idx, score) in enumerate(sim_scores):
            movie = df.iloc[m_idx]
            with cols[i % 3]:
                with st.container(border=True):
                    # Fetching poster from TMDB using IMDb ID (tconst)
                    poster_url = get_poster(movie['tconst'])
                    st.image(poster_url, use_container_width=True)
                    
                    st.markdown(f"**{movie['primaryTitle']}**")
                    st.caption(f"📅 {int(movie['startYear'])} | ⭐ {movie['averageRating']}")

except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.info("Ensure TMDB_API_KEY is added to Streamlit Secrets.")
