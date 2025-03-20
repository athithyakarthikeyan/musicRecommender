import os
import pickle
import difflib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load dataset (Skip fetching from Spotify)
if os.path.exists("spotify_playlist_data.csv"):
    print("\n✅ Using existing playlist data.")
    df = pd.read_csv("spotify_playlist_data.csv")
else:
    print("\n❌ No existing CSV found! Please fetch data first.")
    exit()

# ✅ Ensure 'combined_features' Exists
if "combined_features" not in df.columns:
    print("⚠ 'combined_features' missing! Recomputing...")
    df["artist"] = df["artist"].fillna("")
    df["album"] = df["album"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["name"] = df["name"].fillna("").astype(str)
    df["combined_features"] = df["artist"] + " " + df["album"] + " " + df["genres"]
    df.to_csv("spotify_tracks_preprocessed.csv", index=False)
    print("✅ 'combined_features' column created!")

# ✅ Train KNN Model (If not already trained)
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

# ✅ Compute Cosine Similarity (If not already computed)
if not os.path.exists("similarity_matrix.pkl"):
    print("\n🚀 Computing Cosine Similarity...")
    vectorizer = TfidfVectorizer(stop_words="english")
    feature_matrix = vectorizer.fit_transform(df["combined_features"])
    similarity_matrix = cosine_similarity(feature_matrix)

    with open("similarity_matrix.pkl", "wb") as f:
        pickle.dump(similarity_matrix, f)

    print("✅ Cosine Similarity Computed!")

# ✅ Load Models
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ✅ Find Best Song Match
def find_best_match(input_song):
    df["name"] = df["name"].fillna("").astype(str)
    matches = difflib.get_close_matches(input_song.lower(), df["name"].str.lower(), n=5, cutoff=0.3)
    if not matches:
        return None
    return df[df["name"].str.lower() == matches[0]].iloc[0]

# ✅ Recommend Similar Songs
def recommend_songs(song_input, num_recommendations=5):
    matched_song = find_best_match(song_input)
    if matched_song is None:
        return None, f"❌ No song found matching '{song_input}'. Try again."

    song_vector = vectorizer.transform([matched_song["combined_features"]])
    distances, indices = knn_model.kneighbors(song_vector, n_neighbors=num_recommendations + 1)

    recommended_indices = indices[0][1:num_recommendations + 1]
    recommendations = df.iloc[recommended_indices][["id", "name", "artist", "album", "album_art_url"]]
    return recommendations, None

# ✅ Streamlit Web App
st.title("🎵 Tamil Song Recommendation Engine")
st.write("🔍 Enter a song name to get recommendations.")

with st.form("song_search"):
    song_input = st.text_input("🎶 Enter Song Name:", "")
    submitted = st.form_submit_button("Get Recommendations")

if submitted and song_input:
    recommendations, error_message = recommend_songs(song_input)

    if error_message:
        st.error(error_message)
    else:
        st.success(f"🎶 Showing recommendations for: **{song_input}**")
        for _, row in recommendations.iterrows():
            album_art_url = row.get("album_art_url", "https://via.placeholder.com/150")
            st.image(album_art_url, width=120)
            st.write(f"**{row['name']}** - *{row['artist']}*")

# ✅ Run Streamlit Separately
if __name__ == "__main__":
    print("\n🚀 Run the following command to start the app:")
    print("streamlit run main.py")
