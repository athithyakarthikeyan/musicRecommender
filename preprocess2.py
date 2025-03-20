#
# Algorithm: KNN
#
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

# ✅ Load dataset
df = pd.read_csv("spotify_tracks_preprocessed.csv")

# ✅ Ensure no missing values in important columns
df["artist"] = df["artist"].fillna("")
df["album"] = df["album"].fillna("")
df["genres"] = df["genres"].fillna("")

# ✅ Create 'combined_features' by merging artist, album, and genres
df["combined_features"] = df["artist"] + " " + df["album"] + " " + df["genres"]

# ✅ TF-IDF Vectorization on Combined Features
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(df["combined_features"])

# ✅ Train KNN Model
knn_model = NearestNeighbors(n_neighbors=10, metric="euclidean")
knn_model.fit(feature_matrix)

# ✅ Save Preprocessed Data and KNN Model
df.to_csv("spotify_tracks_preprocessed.csv", index=False)  # ✅ Save processed data

with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Preprocessing Complete! Data and Model Saved Successfully.")

# ✅ Debugging Check
print(df.columns)  # ✅ Ensure 'combined_features' is present
