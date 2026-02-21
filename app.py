import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="CineMatch", page_icon="🍿", layout="wide")

# Custom CSS for a darker, modern look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div.stButton > button:first-child { background-color: #e50914; color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    # IMDb uses 'primaryTitle'. Let's create a 'tags' column using genres.
    df['genres'] = df['genres'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

try:
    df, cosine_sim = load_data()
    st.title("🍿 CineMatch AI")
    st.write("Discover movies and shows based on your favorites.")

    selected_title = st.selectbox("Search for a Movie or TV Show:", df['primaryTitle'].values)

    if st.button('Recommend'):
        idx = df[df['primaryTitle'] == selected_title].index[0]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:7]
        
        st.subheader("Recommended for you:")
        cols = st.columns(3)
        for i, (m_idx, score) in enumerate(sim_scores):
            with cols[i % 3]:
                st.info(f"**{df.iloc[m_idx]['primaryTitle']}**")
                st.caption(f"Year: {df.iloc[m_idx]['startYear']} | Genre: {df.iloc[m_idx]['genres']}")

except Exception as e:
    st.error("Please run export_data.py first to create movies.csv!")