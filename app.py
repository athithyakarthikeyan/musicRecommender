import pandas as pd
import streamlit as st
import difflib
import pickle

# ✅ Load Data
df = pd.read_csv("spotify_tracks_preprocessed.csv")
df["name"] = df["name"].fillna("").astype(str)  # ✅ Ensure names are strings

with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ✅ Find Best Song Match
def find_best_match(input_song):
    """Find the closest matching song"""
    matches = difflib.get_close_matches(input_song.lower(), df["name"].str.lower(), n=5, cutoff=0.3)
    if not matches:
        return None
    return df[df["name"].str.lower() == matches[0]].iloc[0]

# ✅ Recommend Songs
def recommend_songs(song_input, num_recommendations=5):
    matched_song = find_best_match(song_input)
    if matched_song is None:
        return None, "❌ No matching song found!"

    song_vector = vectorizer.transform([matched_song["combined_features"]])
    distances, indices = knn_model.kneighbors(song_vector, n_neighbors=num_recommendations + 1)

    recommendations = df.iloc[indices[0][1:num_recommendations + 1]][["name", "artist", "album", "album_art_url"]]
    return recommendations, None

# ✅ Streamlit Web App
st.title("🎵 Tamil Music Recommender")
song_input = st.text_input("Enter a song name:")

if st.button("Get Recommendations"):
    recommendations, error = recommend_songs(song_input)
    if error:
        st.error(error)
    else:
        for _, row in recommendations.iterrows():
            st.image(row["album_art_url"], width=120)
            st.write(f"**{row['name']}** - *{row['artist']}*")
