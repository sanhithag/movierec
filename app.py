import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- PAGE CONFIG ---
st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")

# This function fetches real posters from TMDB using the IMDb ID (tconst)
def get_poster(imdb_id):
    # Using a public educational API key
    url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&external_source=imdb_id"
    try:
        data = requests.get(url).json()
        poster_path = data['movie_results'][0]['poster_path'] if data['movie_results'] else data['tv_results'][0]['poster_path']
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

# --- DATA ENGINE ---
@st.cache_data
def load_data():
    # Use low_memory=False for faster, more stable loading of large CSVs
    df = pd.read_csv('movies.csv', low_memory=False)
    
    # Fill empty genres once
    df['genres'] = df['genres'].fillna('')
    
    # Standard TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])
    
    # Linear kernel is memory-efficient for this scale
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

# --- UI ---
try:
    df, cosine_sim = load_data()
    st.title("🎬 CineMatch AI")
    st.write("Find your next favorite movie or TV show.")

    # Search bar using the titles from your exported IMDb data
    selected_movie = st.selectbox("Type a movie/show you like:", df['primaryTitle'].values)

    if st.button('Get Recommendations'):
        idx = df[df['primaryTitle'] == selected_movie].index[0]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
        
        st.subheader(f"Because you liked {selected_movie}...")
        cols = st.columns(5)
        for i, (m_idx, score) in enumerate(sim_scores):
            movie_data = df.iloc[m_idx]
            with cols[i]:
                # Fetching the poster using the tconst column from movies.csv
                st.image(get_poster(movie_data['tconst']), use_container_width=True)
                st.write(f"**{movie_data['primaryTitle']}**")
                st.caption(f"⭐ {movie_data['averageRating']} | {int(movie_data['startYear'])}")

except Exception as e:
    st.error("The app is currently loading the dataset. Please refresh in a moment.")

