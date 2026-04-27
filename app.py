import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="PlotMatch AI", page_icon="🎬", layout="wide")

# 2. CUSTOM UI STYLING (The "Non-AI" look)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .movie-card {
        background: #1e1e26;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #333;
        transition: transform 0.3s ease;
        height: 620px;
    }
    .movie-card:hover {
        transform: translateY(-8px);
        border-color: #ff4b4b;
    }
    .plot-text {
        color: #999;
        font-size: 0.85rem;
        height: 80px;
        overflow: hidden;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. DATA & SIMILARITY ENGINE
@st.cache_resource # Use resource for large math objects
def load_and_compute():
    # Load the lite file you created
    df = pd.read_pickle("movies_lite.pkl")
    
    # Vectorize the plot overviews
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    
    # Calculate Similarity Scores
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, similarity

try:
    df, similarity = load_and_compute()
except FileNotFoundError:
    st.error("Missing 'movies_lite.pkl'. Please run your preprocessing script first!")
    st.stop()

# 4. API HELPER
def get_omdb_details(movie_title):
    api_key = st.secrets["OMDB_API_KEY"]
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    try:
        return requests.get(url).json()
    except:
        return None

# 5. RECOMMENDATION LOGIC
def recommend(movie_title):
    try:
        idx = df[df['title'] == movie_title].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        
        results = []
        for i in distances[1:7]: # Get top 6 matches
            results.append(df.iloc[i[0]].title)
        return results
    except Exception:
        return []

# 6. APP INTERFACE
st.title("🧠 PlotMatch AI")
st.markdown("##### Discover movies based on plot similarity, not just genres.")

# Search bar
selected_movie = st.selectbox(
    "Type or select a movie you liked:",
    df['title'].values,
    index=None,
    placeholder="Select a movie..."
)

if selected_movie:
    with st.spinner("Finding similar stories..."):
        recommendations = recommend(selected_movie)
        
        if recommendations:
            st.write("---")
            cols = st.columns(3)
            
            for idx, title in enumerate(recommendations):
                movie_info = get_omdb_details(title)
                
                with cols[idx % 3]:
                    # Visuals
                    poster = movie_info.get('Poster', "") if movie_info else ""
                    if poster == "N/A" or not poster:
                        poster = "https://via.placeholder.com/300x450?text=No+Poster"
                    
                    plot = movie_info.get('Plot', "No description available.") if movie_info else "Description not found."
                    rating = movie_info.get('imdbRating', 'N/A') if movie_info else "N/A"

                    # Displaying the Card
                    st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster}" style="width:100%; border-radius:8px;">
                            <h3 style="color:white; font-size:1.1rem; margin-top:10px;">{title}</h3>
                            <p style="color:#ff4b4b; font-weight:bold;">⭐ {rating}</p>
                            <div class="plot-text">{plot}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.write("") # Margin
        else:
            st.warning("We couldn't find matches for that one.")

# 7. SIDEBAR & CREDITS
st.sidebar.header("How it works")
st.sidebar.write("""
This engine uses **Natural Language Processing (NLP)**. 
It converts movie plots into mathematical vectors and uses **Cosine Similarity** to find the 'distance' between stories.
""")
st.sidebar.caption("Data: TMDB & OMDb")
