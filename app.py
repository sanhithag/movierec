import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="CineMatch", page_icon="🍿", layout="wide")

@st.cache_data
def load_and_prep_data():
    # Load and filter immediately to save RAM
    df = pd.read_csv('movies.csv', low_memory=False)
    
    # Clean data: Filter for movies > 2000 and ensure ratings exist
    df = df[df['startYear'] > 2000].copy()
    df['genres'] = df['genres'].fillna('')
    
    # IMPORTANT: Reset index so df index matches tfidf_matrix rows
    df = df.reset_index(drop=True)
    return df

@st.cache_resource # Use cache_resource for objects like the TF-IDF matrix
def compute_tfidf(genres_series):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(genres_series)
    return matrix

# --- App Logic ---

try:
    df = load_and_prep_data()
    tfidf_matrix = compute_tfidf(df['genres'])

    st.title("🍿 CineMatch AI")
    
    # Using a selectbox with a search feature
    selected_title = st.selectbox("Search for a Movie/Show:", df['primaryTitle'].values)

    if st.button('Recommend'):
        # Find index of selected movie
        idx = df[df['primaryTitle'] == selected_title].index[0]
        
        # Calculate similarity for just this row
        cosine_sim_single = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Get top matches (excluding the movie itself)
        sim_scores = sorted(list(enumerate(cosine_sim_single)), key=lambda x: x[1], reverse=True)[1:7]
        
        st.subheader(f"Because you liked '{selected_title}':")
        cols = st.columns(3)
        
        for i, (m_idx, score) in enumerate(sim_scores):
            movie = df.iloc[m_idx]
            with cols[i % 3]:
                with st.container(border=True): # Adds a nice frame
                    st.markdown(f"**{movie['primaryTitle']}**")
                    st.caption(f"📅 {int(movie['startYear'])} | ⭐ {movie['averageRating']}")

except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.info("Check if 'movies.csv' is in the same folder and has the correct columns.")
