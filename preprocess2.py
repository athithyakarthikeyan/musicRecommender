import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ✅ Load dataset
df = pd.read_csv("spotify_playlist_data.csv")

# ✅ Fix missing song names
df["name"] = df["name"].fillna("").astype(str)
df["artist"] = df["artist"].fillna("")
df["album"] = df["album"].fillna("")
df["genres"] = df["genres"].fillna("")

# ✅ Ensure 'combined_features' exists
df["combined_features"] = df["artist"] + " " + df["album"] + " " + df["genres"]

# ✅ Train KNN Model
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(df["combined_features"])

knn_model = NearestNeighbors(n_neighbors=10, metric="euclidean")
knn_model.fit(feature_matrix)

# ✅ Save processed data
df.to_csv("spotify_tracks_preprocessed.csv", index=False)
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ KNN Model Training Complete!")
