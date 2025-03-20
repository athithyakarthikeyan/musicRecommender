import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ✅ Load dataset
df = pd.read_csv("spotify_playlist_data.csv")

# ✅ Ensure 'combined_features' Exists
if "combined_features" not in df.columns:
    df["artist"] = df["artist"].fillna("")
    df["album"] = df["album"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["name"] = df["name"].fillna("").astype(str)
    df["combined_features"] = df["artist"] + " " + df["album"] + " " + df["genres"]
    df.to_csv("spotify_tracks_preprocessed.csv", index=False)

# ✅ Check if model exists, otherwise train it
if not os.path.exists("knn_model.pkl"):
    print("\n🚀 Training KNN Model...")
    vectorizer = TfidfVectorizer(stop_words="english")
    feature_matrix = vectorizer.fit_transform(df["combined_features"])

    knn_model = NearestNeighbors(n_neighbors=10, metric="euclidean")
    knn_model.fit(feature_matrix)

    with open("knn_model.pkl", "wb") as f:
        pickle.dump(knn_model, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ KNN Model Training Complete!")

# ✅ Load Models
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ✅ Streamlit Web App
st.title("🎵 Tamil Song Recommendation Engine")
st.write("🔍 Enter a song name to get recommendations.")

with st.form("song_search"):
    song_input = st.text_input("🎶 Enter Song Name:", "")
    submitted = st.form_submit_button("Get Recommendations")

if submitted and song_input:
    st.success(f"🎶 Showing recommendations for: **{song_input}**")
