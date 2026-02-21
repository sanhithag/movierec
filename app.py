import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_data
def load_and_clean_data():
    # Load the actual CSV
    df = pd.read_csv('movies.csv')
    
    # Fill empty values so the vectorizer doesn't crash
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('')
    
    # Combine features for better matching
    df['tags'] = df['overview'] + " " + df['genres']
    return df

try:
    df = load_and_clean_data()

    # Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    
    # linear_kernel is faster than cosine_similarity for TF-IDF
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    st.title("🎬 Data-Driven Movie Recommender")

    movie_list = df['title'].values
    selected_movie = st.selectbox("Type or select a movie:", movie_list)

    if st.button('Show Recommendation'):
        idx = df[df['title'] == selected_movie].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 5
        movie_indices = [i[0] for i in sim_scores[1:6]]
        
        cols = st.columns(5)
        for i, col in enumerate(cols):
            col.write(df.iloc[movie_indices[i]]['title'])
            
except FileNotFoundError:
    st.error("❌ 'movies.csv' not found! Make sure the file is in the same folder as app.py")
except Exception as e:
    st.error(f"⚠️ Error: {e}")
