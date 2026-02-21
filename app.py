import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Movierecs cuz im lazy ash", page_icon="🍿", layout="wide")

# --- IMPROVED DEEP SEARCH HELPERS ---
def get_movie_details(imdb_id, title):
    api_key = st.secrets["TMDB_API_KEY"]
    
    # TRY 1: Find by exact IMDb ID
    find_url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
    try:
        data = requests.get(find_url).json()
        if data.get('movie_results'):
            res = data['movie_results'][0]
            return res['id'], res['overview'], f"https://image.tmdb.org/t/p/w500{res['poster_path']}", "movie"
        if data.get('tv_results'):
            res = data['tv_results'][0]
            return res['id'], res['overview'], f"https://image.tmdb.org/t/p/w500{res['poster_path']}", "tv"
            
        # TRY 2: Backup search by Title (if the ID handshake fails)
        search_url = f"https://api.themoviedb.org/3/search/multi?api_key={api_key}&query={title}"
        search_res = requests.get(search_url).json()
        if search_res.get('results'):
            res = search_res['results'][0]
            m_type = res.get('media_type', 'movie')
            poster = f"https://image.tmdb.org/t/p/w500{res.get('poster_path')}" if res.get('poster_path') else "https://via.placeholder.com/500"
            return res['id'], res['overview'], poster, m_type
    except:
        return None, None, None, None
    return None, None, None, None

def get_recommendations(tmdb_id, media_type="movie"):
    api_key = st.secrets["TMDB_API_KEY"]
    # TMDB's internal AI matches stories/keywords
    rec_url = f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}/recommendations?api_key={api_key}"
    try:
        return requests.get(rec_url).json().get('results', [])[:6]
    except:
        return []

# --- DATA ENGINE ---
@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv', low_memory=False)
    # WIDEN THE NET: Only filter out things with almost zero votes
    df = df[df['numVotes'] > 500].copy()
    return df.sort_values('primaryTitle')

# --- APP UI ---
try:
    df = load_data()
    st.title("🍿 Plot-Based Recommender")

    selected_title = st.selectbox("Select a movie/show:", df['primaryTitle'].values)

    if st.button('Find Similar Stories'):
        row = df[df['primaryTitle'] == selected_title].iloc[0]
        tmdb_id, plot, poster, m_type = get_movie_details(row['tconst'], selected_title)
        
        if tmdb_id:
            with st.expander("📌 Original Plot Summary"):
                col1, col2 = st.columns([1, 4])
                col1.image(poster)
                col2.write(plot)
            
            st.markdown("---")
            recs = get_recommendations(tmdb_id, m_type)
            
            if recs:
                cols = st.columns(3)
                for i, movie in enumerate(recs):
                    with cols[i % 3]:
                        with st.container(border=True):
                            m_title = movie.get('title') or movie.get('name')
                            p_path = movie.get('poster_path')
                            img = f"https://image.tmdb.org/t/p/w500{p_path}" if p_path else "https://via.placeholder.com/500"
                            st.image(img, use_container_width=True)
                            st.markdown(f"**{m_title}**")
                            st.caption(f"{movie['overview'][:150]}...")
            else:
                st.warning("No plot matches found. Try another movie!")
        else:
            st.error("This title is missing from the plot database.")

except Exception as e:
    st.error(f"Error: {e}")
