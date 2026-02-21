import streamlit as st
import pandas as pd
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="Movierecs", page_icon="🎞️", layout="wide")

# --- TMDB API HELPERS ---
def get_movie_details(imdb_id):
    api_key = st.secrets["TMDB_API_KEY"]
    # 1. Find the TMDB ID from the IMDb ID (tconst)
    find_url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
    try:
        data = requests.get(find_url).json()
        if data['movie_results']:
            movie = data['movie_results'][0]
            return movie['id'], movie['overview'], f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
        elif data['tv_results']:
            tv = data['tv_results'][0]
            return tv['id'], tv['overview'], f"https://image.tmdb.org/t/p/w500{tv['poster_path']}"
    except:
        return None, None, None

def get_recommendations(tmdb_id, media_type="movie"):
    api_key = st.secrets["TMDB_API_KEY"]
    # 2. Get recommendations based on plot keywords/stories
    rec_url = f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}/recommendations?api_key={api_key}"
    try:
        return requests.get(rec_url).json().get('results', [])[:6]
    except:
        return []

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv', low_memory=False)
    # We'll stick to popular modern movies to ensure the API has data for them
    df = df[(df['startYear'] > 2000) & (df['numVotes'] > 20000)]
    return df.sort_values('primaryTitle')

# --- UI ---
try:
    df = load_data()
    st.title("Plot-recs")
    st.write("Search for a movie and get similar movie recs. Yay!!!!")

    selected_title = st.selectbox("Select a movie you love:", df['primaryTitle'].values)

    if st.button('Find Similar Stories'):
        # Get the IMDb ID from our CSV
        imdb_id = df[df['primaryTitle'] == selected_title]['tconst'].values[0]
        
        tmdb_id, original_plot, original_poster = get_movie_details(imdb_id)
        
        if tmdb_id:
            # Show the "Source" movie plot
            with st.expander("Show Original Plot Summary"):
                col_a, col_b = st.columns([1, 4])
                col_a.image(original_poster)
                col_b.write(original_plot)
            
            st.markdown("---")
            st.subheader("Recommended for the Plot:")
            
            # Fetch recommendations from the API
            recs = get_recommendations(tmdb_id)
            
            if recs:
                cols = st.columns(3)
                for i, movie in enumerate(recs):
                    with cols[i % 3]:
                        with st.container(border=True):
                            poster = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie['poster_path'] else "https://via.placeholder.com/500"
                            st.image(poster, use_container_width=True)
                            st.markdown(f"**{movie.get('title', movie.get('name'))}**")
                            # Show the plot snippet
                            st.caption(f"{movie['overview'][:150]}...")
            else:
                st.warning("No specific plot matches found. Try another movie!")
        else:
            st.error("Could not find this movie in the plot database.")

except Exception as e:
    st.error(f"Error: {e}")

