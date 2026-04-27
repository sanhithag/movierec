import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="movierec", page_icon="🎬", layout="wide")

# 2. API Key Setup
# Get your free key at http://www.omdbapi.com/apikey.aspx
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

# 3. Custom CSS (Fixed and Polished)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
        border-radius: 10px;
        border: 1px solid #444;
    }
    .movie-card {
        background: #1e1e26;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        border: 1px solid #333;
        transition: transform 0.3s;
        height: 520px; /* Uniform height for grid */
    }
    .movie-card:hover {
        transform: translateY(-10px);
        border-color: #ff4b4b;
    }
    .poster {
        border-radius: 10px;
        margin-bottom: 10px;
        object-fit: cover;
    }
    .title-text {
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 5px 0;
    }
    .meta-text {
        color: #999;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 4. Data Fetching Logic
def search_movies(title):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&s={title}"
    try:
        response = requests.get(url).json()
        if response.get("Response") == "True":
            return response.get("Search")[:6] # Top 6 results
    except Exception as e:
        st.error(f"Connection Error: {e}")
    return None

# 5. UI Layout
st.title("movierec")
st.write("A movie reccommendation system using OMDb")

query = st.text_input("", placeholder="Search for a movie (e.g., Interstellar)")

if query:
    results = search_movies(query)
    if results:
        cols = st.columns(3) # 3-column grid
        for i, movie in enumerate(results):
            with cols[i % 3]:
                # Poster handling
                img = movie['Poster'] if movie['Poster'] != "N/A" else "https://via.placeholder.com/300x450?text=No+Poster"
                
                # Using HTML for the card to ensure it looks "Custom"
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{img}" class="poster" width="100%">
                        <div class="title-text">{movie['Title']}</div>
                        <div class="meta-text">{movie['Year']} • {movie['Type'].capitalize()}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.write("") # Spacer
    else:
        st.warning("No movies found. Try another title!")

# 6. Sidebar Credit (Required by most APIs)
st.sidebar.title("About")
st.sidebar.info("This project showcases API integration and Custom CSS in Streamlit.")
st.sidebar.caption("Data source: OMDb API")
