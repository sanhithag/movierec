import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Movierecs", page_icon="🎞️", layout="wide")

def get_movie_details(imdb_id, title):
    api_key = st.secrets["TMDB_API_KEY"]
    # We add 'append_to_response' to get providers and ratings in one call
    find_url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
    
    try:
        data = requests.get(find_url).json()
        res = None
        m_type = "movie"
        
        if data.get('movie_results'):
            res = data['movie_results'][0]
        elif data.get('tv_results'):
            res = data['tv_results'][0]
            m_type = "tv"

        if res:
            tmdb_id = res['id']
            # Fetch Watch Providers (OTT)
            ott_url = f"https://api.themoviedb.org/3/{m_type}/{tmdb_id}/watch/providers?api_key={api_key}"
            ott_data = requests.get(ott_url).json()
            # Defaulting to 'US' region, change to 'IN' or your country code
            providers = ott_data.get('results', {}).get('US', {}).get('flatrate', [])
            ott_list = [p['provider_name'] for p in providers]
            
            poster = f"https://image.tmdb.org/t/p/w500{res['poster_path']}" if res.get('poster_path') else "https://via.placeholder.com/500"
            return tmdb_id, res['overview'], poster, m_type, res.get('vote_average'), ott_list
            
    except Exception as e:
        return None, None, None, None, None, []
    return None, None, None, None, None, []

# --- APP UI ---
try:
    df = load_data() # (Assuming load_data is defined as before)
    st.title("Plot-based recs + Streaming Info")

    selected_title = st.selectbox("Type a movie/tv show", df['primaryTitle'].values)

    if st.button('Find Similar recs'):
        row = df[df['primaryTitle'] == selected_title].iloc[0]
        # Get extra data: rating and ott_platforms
        tmdb_id, plot, poster, m_type, rating, ott = get_movie_details(row['tconst'], selected_title)
        
        if tmdb_id:
            with st.expander("📌 Original Plot & Info"):
                col1, col2 = st.columns([1, 4])
                col1.image(poster)
                col2.write(f"**Rating:** ⭐ {rating}/10")
                if ott:
                    col2.write(f"**Available on:** {', '.join(ott)}")
                else:
                    col2.write("**Available on:** Not found (Check JustWatch)")
                col2.write(plot)
            
            st.markdown("---")
            recs = get_recommendations(tmdb_id, m_type)
            
            if recs:
                cols = st.columns(3)
                for i, movie in enumerate(recs):
                    with cols[i % 3]:
                        with st.container(border=True):
                            m_title = movie.get('title') or movie.get('name')
                            st.image(f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}", use_container_width=True)
                            st.markdown(f"**{m_title}**")
                            st.caption(f"⭐ {movie.get('vote_average')} | {movie['overview'][:100]}...")
