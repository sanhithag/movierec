import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Setup Page
st.set_page_config(page_title="Movie Matcher", layout="wide")
st.title("🎬 Movie Recommendation System")

# 2. Load Data (Using a sample dataset)
@st.cache_data
def load_data():
    # In a real scenario, you'd use the TMDB 5000 dataset.
    # For now, let's create a small internal sample to ensure the code runs.
    data = {
        'title': ['The Dark Knight', 'Inception', 'Toy Story', 'Finding Nemo', 'Interstellar'],
        'genres': ['Action Crime Drama', 'Action Adventure Sci-Fi', 'Animation Adventure Comedy', 'Animation Adventure Comedy', 'Adventure Drama Sci-Fi'],
        'overview': ['Batman fights Joker', 'Dreams within dreams', 'Toys come to life', 'Fish finds son', 'Space exploration and time']
    }
    df = pd.DataFrame(data)
    # Combine features into one "tags" column
    df['tags'] = df['genres'] + " " + df['overview']
    return df

df = load_data()

# 3. Vectorization and Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])
similarity = cosine_similarity(tfidf_matrix)

# 4. Recommendation Logic
def recommend(movie_title):
    try:
        idx = df[df['title'] == movie_title].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        
        # Get top 3 (excluding itself)
        recs = [df.iloc[i[0]].title for i in distances[1:4]]
        return recs
    except:
        return ["Movie not found in database!"]

# 5. UI Layout
selected_movie = st.selectbox("Search for a movie:", df['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.write("### You might also like:")
    for i in recommendations:
        st.success(i)