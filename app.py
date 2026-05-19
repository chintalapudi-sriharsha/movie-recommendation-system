import pandas as pd
import ast
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# STEP 1: LOAD DATA (CACHED)
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    credits = pd.read_csv("credits.csv.zip", compression='zip')
    return movies, credits

movies, credits = load_data()

# -----------------------------
# STEP 2: MERGE DATA
# -----------------------------
movies = movies.merge(credits, on="title")

movies = movies[[
    "movie_id",
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew"
]]

movies.dropna(inplace=True)

# -----------------------------
# STEP 3: DATA CLEANING
# -----------------------------
def convert(text):
    return [i["name"] for i in ast.literal_eval(text)]

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)

def convert_cast(text):
    L = []
    for i, item in enumerate(ast.literal_eval(text)):
        if i < 3:
            L.append(item["name"])
    return L

movies["cast"] = movies["cast"].apply(convert_cast)

def fetch_director(text):
    return [i["name"] for i in ast.literal_eval(text) if i["job"] == "Director"]

movies["crew"] = movies["crew"].apply(fetch_director)

def collapse(L):
    return [i.replace(" ", "") for i in L]

movies["genres"] = movies["genres"].apply(collapse)
movies["keywords"] = movies["keywords"].apply(collapse)
movies["cast"] = movies["cast"].apply(collapse)
movies["crew"] = movies["crew"].apply(collapse)

# -----------------------------
# STEP 4: CREATE TAGS
# -----------------------------
movies["tags"] = (
    movies["overview"].astype(str) + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x)) + " " +
    movies["cast"].apply(lambda x: " ".join(x)) + " " +
    movies["crew"].apply(lambda x: " ".join(x))
)

movies = movies[["movie_id", "title", "tags"]]
movies["tags"] = movies["tags"].apply(lambda x: x.lower())

movies = movies.reset_index(drop=True)

# -----------------------------
# STEP 5: ML MODEL (CACHED)
# -----------------------------
@st.cache_data
def build_model(data):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data['tags'])
    similarity = cosine_similarity(matrix)
    return similarity

similarity = build_model(movies)

# -----------------------------
# STEP 6: RECOMMEND FUNCTION
# -----------------------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]

    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, reverse=True, key=lambda x: x[1])

    return [movies.iloc[i[0]]['title'] for i in distances[1:6]]

# -----------------------------
# STEP 7: STREAMLIT UI
# -----------------------------
st.title("🎬 Movie Recommendation System")

st.write("Select a movie and get similar recommendations")

movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend"):
    results = recommend(movie)

    st.subheader("Recommended Movies:")
    for r in results:
        st.write("👉", r)