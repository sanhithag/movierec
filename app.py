import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="CineMatch Pro", page_icon="🍿", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    df['genres'] = df['genres'].fillna('')
    # Pre-calculate similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

try:
    df, cosine_sim = load_data()

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter Content")
    content_type = st.sidebar.multiselect("Select Type:", options=df['titleType'].unique(), default=df['titleType'].unique())
    
    # Filter the dataframe based on sidebar
    filtered_df = df[df['titleType'].isin(content_type)]

    # --- MAIN UI ---
    st.title("🍿 CineMatch AI")
    selected_title = st.selectbox("Search for a Movie or TV Show:", filtered_df['primaryTitle'].values)

    if st.button('Get Recommendations'):
        idx = df[df['primaryTitle'] == selected_title].index[0]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:10]
        
        st.subheader("Results:")
        cols = st.columns(3)
        for i, (m_idx, score) in enumerate(sim_scores):
            movie_row = df.iloc[m_idx]
            with cols[i % 3]:
                st.info(f"**{movie_row['primaryTitle']}**")
                st.write(f"⭐ Rating: {movie_row['averageRating']} | 📅 {int(movie_row['startYear'])}")
                st.caption(f"Type: {movie_row['titleType']} | {movie_row['genres']}")

except Exception as e:
    st.error(f"Waiting for data... Ensure app.py and movies.csv are in your GitHub root. Error: {e}")
