import streamlit as st
import pandas as pd
import requests

# 1. Load your local (pre-filtered) IMDb data
@st.cache_data
def load_data():
    return pd.read_csv("movies_lite.csv") # You created this from the IMDb TSV

df = load_data()

# 2. Function to get posters using the IMDb ID (tconst)
def get_poster_url(imdb_id):
    # OMDb API is a great TMDb alternative for posters
    api_key = st.secrets["OMDB_API_KEY"]
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
    try:
        data = requests.get(url).json()
        return data.get('Poster', "https://via.placeholder.com/300x450")
    except:
        return "https://via.placeholder.com/300x450"

# --- UI ---
st.title("IMDb Local Explorer")
query = st.text_input("Search a movie from local database:")

if query:
    results = df[df['primaryTitle'].str.contains(query, case=False)].head(3)
    
    cols = st.columns(3)
    for i, (index, row) in enumerate(results.iterrows()):
        with cols[i]:
            poster = get_poster_url(row['tconst'])
            st.image(poster, caption=row['primaryTitle'])
            st.write(f"Rating: ⭐ {row['averageRating']}")
