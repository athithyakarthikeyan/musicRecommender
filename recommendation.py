#
# Algorithm: Cosine similarity on song metadata and Spotify API
#
import pandas as pd
import difflib
import pickle

# ✅ Corrected file paths (remove "/mnt/data/")
df = pd.read_csv("spotify_playlist_data.csv")  # Ensure correct file path
with open("similarity_matrix.pkl", "rb") as f:
    similarity_matrix = pickle.load(f)

# ✅ Function to Find Closest Song Matches
def find_best_match(input_song):
    matches = difflib.get_close_matches(input_song.lower(), df["name"].str.lower(), n=5, cutoff=0.3)
    if matches:
        return df[df["name"].str.lower() == matches[0]]["name"].values[0]
    return None

# ✅ Recommendation Function with Partial Matching
def recommend_songs(song_input, num_recommendations=5):
    matched_song = find_best_match(song_input)
    if matched_song is None:
        print(f"❌ No song found matching '{song_input}'. Try again.")
        return None

    print(f"🔍 Searching recommendations for: {matched_song}")

    # Find index of the matched song
    song_idx = df[df["name"] == matched_song].index[0]

    # Get similarity scores for all songs
    similarity_scores = list(enumerate(similarity_matrix[song_idx]))

    # Sort songs by similarity score (highest first), ignore the first song (itself)
    sorted_songs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

    # Return recommended songs
    recommendations = df.iloc[[i[0] for i in sorted_songs]][["name", "artist", "album"]]
    return recommendations

# ✅ Run the Recommendation Engine
if __name__ == "__main__":
    song_input = input("Enter part of a song name: ").strip()
    recommended_songs = recommend_songs(song_input)

    if recommended_songs is not None:
        print("\n🎶 Recommended Songs:")
        print(recommended_songs)
    else:
        print("❌ No recommendations found.")
