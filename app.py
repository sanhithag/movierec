import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="CineMatch", page_icon="🍿", layout="wide")

@st.cache_data
def load_data():
    # Load data and keep only the columns we need to save RAM
    df = pd.read_csv('movies.csv', usecols=['tconst', 'primaryTitle', 'genres', 'averageRating', 'startYear'])
    df['genres'] = df['genres'].fillna('')
    return df

try:
    df = load_data()
    
    # We move the Vectorizer OUTSIDE the load_data to process it only when needed
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])

    st.title("🍿 CineMatch AI")
    selected_title = st.selectbox("Search for a Movie/Show:", df['primaryTitle'].values)

    if st.button('Recommend'):
        # We calculate similarity ONLY for the selected movie to save memory
        idx = df[df['primaryTitle'] == selected_title].index[0]
        # Instead of a full matrix, we just compare the selected movie to everything else
        cosine_sim_single = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Get top 6 matches
        sim_scores = sorted(list(enumerate(cosine_sim_single)), key=lambda x: x[1], reverse=True)[1:7]
        
        st.subheader("Recommended for you:")
        cols = st.columns(3)
        for i, (m_idx, score) in enumerate(sim_scores):
            with cols[i % 3]:
                st.info(f"**{df.iloc[m_idx]['primaryTitle']}**")
                st.caption(f"Year: {df.iloc[m_idx]['startYear']} | ⭐ {df.iloc[m_idx]['averageRating']}")

except Exception as e:
    st.error("App is initializing. Please wait 30 seconds and refresh.")
