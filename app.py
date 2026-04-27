import streamlit as st
import requests

# Page Config for a better title/icon in the browser tab
st.set_page_config(page_title="Cinematch", page_icon="🍿", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }
    
    /* Movie Card Styling */
    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 100%;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 10px 20px rgba(0,0,0,0.4);
        border: 1px solid #ff4b4b;
    }

    /* Title Styling */
    .movie-title {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_input=True)

# --- LOGIC ---
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

def get_recommendations(movie_title):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(search_url).json()
    if response['results']:
        movie_id = response['results'][0]['id']
        rec_url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={TMDB_API_KEY}"
        return requests.get(rec_url).json()['results'][:6] # 6 works best for a 3-column grid
    return []

# --- UI ---
st.title("🍿 Cinematch")
st.markdown("##### Discover your next favorite film without the scrolling fatigue.")

user_input = st.text_input("", placeholder="Type a movie you love (e.g., Inception)...")

if user_input:
    recs = get_recommendations(user_input)
    if recs:
        # Create a grid layout
        cols = st.columns(3) 
        for idx, movie in enumerate(recs):
            with cols[idx % 3]:
                # Injecting HTML for the custom card look
                poster = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie['poster_path'] else ""
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster}" style="width:100%; border-radius:10px;">
                        <div class="movie-title">{movie['title']}</div>
                        <p style="color: #888; font-size: 0.8rem;">⭐ {movie['vote_average']} | {movie['release_date'][:4]}</p>
                    </div>
                """, unsafe_allow_input=True)
                st.write("") # Spacer
    else:
        st.warning("We couldn't find that one. Try checking the spelling!")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("Data provided by [TMDb](https://www.themoviedb.org/)")
