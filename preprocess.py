#
# Algorithm: Cosine similarity on song metadata and Spotify API
#

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ✅ Load the original dataset
df = pd.read_csv("spotify_playlist_data.csv")  # ✅ Use the correct CSV file


# ✅ Keep album_art_url in the processed data
album_art_urls = df["album_art_url"]

# ✅ Normalize numerical columns (popularity, duration)
scaler = MinMaxScaler()
df[["popularity", "duration_ms"]] = scaler.fit_transform(df[["popularity", "duration_ms"]])

# ✅ Convert genres to numerical format using TF-IDF
vectorizer = TfidfVectorizer()
df["genres"] = df["genres"].fillna("")
genre_matrix = vectorizer.fit_transform(df["genres"])

# ✅ Compute similarity matrices
numeric_features = df[["popularity", "duration_ms"]].values
numeric_similarity = cosine_similarity(numeric_features)
genre_similarity = cosine_similarity(genre_matrix)

# ✅ Combine similarity scores
similarity_matrix = (numeric_similarity * 0.5) + (genre_similarity * 0.5)

# ✅ Add album_art_url back
df["album_art_url"] = album_art_urls

# ✅ Save preprocessed data
df.to_csv("spotify_tracks_preprocessed.csv", index=False)  # ✅ Save processed data

with open("similarity_matrix.pkl", "wb") as f:
    pickle.dump(similarity_matrix, f)

print("✅ Preprocessing Complete! Data and similarity matrix saved.")
