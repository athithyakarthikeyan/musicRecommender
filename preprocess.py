import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ✅ Load dataset
df = pd.read_csv("spotify_playlist_data.csv")

# ✅ Fix missing song names
df["name"] = df["name"].fillna("").astype(str)
df["artist"] = df["artist"].fillna("")
df["album"] = df["album"].fillna("")
df["genres"] = df["genres"].fillna("")

# ✅ Ensure 'combined_features' exists
df["combined_features"] = df["artist"] + " " + df["album"] + " " + df["genres"]

# ✅ Compute Cosine Similarity
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(df["combined_features"])
similarity_matrix = cosine_similarity(feature_matrix)

# ✅ Save processed data
df.to_csv("spotify_tracks_preprocessed.csv", index=False)
with open("similarity_matrix.pkl", "wb") as f:
    pickle.dump(similarity_matrix, f)

print("✅ Preprocessing complete! Cosine similarity matrix saved.")
