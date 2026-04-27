import streamlit as st
import requests

# Page Setup
st.set_page_config(page_title="FilmFinder", page_icon="🎬", layout="wide")

# OMDb API Key (Get yours at http://www.omdbapi.com/apikey.aspx)
# Add this to your Streamlit Secrets as OMDB_API_KEY
API_KEY = st.secrets["OMDB_API_KEY"]

# Custom CSS for a "Premium" feel
st.markdown("""
    <style>
    .main { background-color: #050505; }
    .movie-card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 20px;
        border: 1px solid #333;
        transition: all 0.3s ease;
    }
    .movie-card:hover {
        border-color: #00d4ff;
        transform: scale(1.02);
    }
    .movie-title {
        color: white;
        font-weight: bold;
        font-size: 1rem;
        margin-top: 8px;
    }
    .movie-meta {
        color: #00d4ff;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_input=True)

def fetch_movie_data(title):
    # OMDb uses 's' for search (returns list) or 't' for title (returns one)
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&s={title}"
    response = requests.get(url).json()
    if response.get("Response") == "True":
        return response.get("Search")[:6] # Return first 6 results
    return None

# UI Layout
st.title("🎬 FilmFinder")
st.write("Search for movies to see a modern recommendation-style layout.")

query = st.text_input("", placeholder="Enter a movie name...")

if query:
    results = fetch_movie_data(query)
    if results:
        cols = st.columns(3)
        for i, movie in enumerate(results):
            with cols[i % 3]:
                # Handling missing posters
                poster_url = movie['Poster'] if movie['Poster'] != "N/A" else "https://via.placeholder.com/300x450?text=No+Image"
                
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster_url}" style="width:100%; border-radius:8px;">
                        <div class="movie-title">{movie['Title']}</div>
                        <div class="movie-meta">{movie['Year']} • {movie['Type'].capitalize()}</div>
                    </div>
                """, unsafe_allow_input=True)
    else:
        st.error("No results found. Please check your connection or API key.")

st.sidebar.info("Using OMDb API as a TMDb alternative.")
