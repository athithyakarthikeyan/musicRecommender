import streamlit as st
import pandas as pd
import pickle
import difflib
import uuid

# ✅ Load Preprocessed Data and KNN Model
df = pd.read_csv("spotify_playlist_data.csv")
if "combined_features" not in df.columns:
    print("⚠ 'combined_features' missing! Recomputing...")
    df["artist"] = df["artist"].fillna("")
    df["album"] = df["album"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["name"] = df["name"].fillna("").astype(str)
    
    df["combined_features"] = df["artist"] + " " + df["album"] + " " + df["genres"]
    df.to_csv("spotify_tracks_preprocessed.csv", index=False)  # ✅ Save the fixed dataset

    print("✅ 'combined_features' column created!")

# ✅ Ensure 'combined_features' is available
if "combined_features" not in df.columns:
    st.error("❌ Error: 'combined_features' is missing from the dataset. Please re-run preprocess.py.")
    st.stop()

with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ✅ Function to Find Closest Song Matches
def find_best_match(input_song):
    matches = difflib.get_close_matches(input_song.lower(), df["name"].str.lower(), n=5, cutoff=0.3)
    if matches:
        return df[df["name"].str.lower() == matches[0]]
    return None

# ✅ KNN-Based Recommendation Function
def recommend_songs(song_input, num_recommendations=10):
    matched_song = find_best_match(song_input)
    if matched_song is None or matched_song.empty:
        return None, None, f"❌ No song found matching '{song_input}'. Try again."

    song_idx = matched_song.index[0]

    # ✅ Transform Input Song into TF-IDF Vector
    song_vector = vectorizer.transform([df.iloc[song_idx]["combined_features"]])

    # ✅ Find KNN Neighbors
    distances, indices = knn_model.kneighbors(song_vector, n_neighbors=num_recommendations + 1)

    # ✅ Get Recommended Songs (Ignoring First, Which is Itself)
    recommended_indices = indices[0][1:num_recommendations + 1]

    recommendations = df.iloc[recommended_indices][["id", "name", "artist", "album", "album_art_url"]]
    return matched_song.iloc[0], recommendations, None  # Return the exact match and recommendations

# ✅ Initialize session state for search input and recommendations
if "selected_track_id" not in st.session_state:
    st.session_state["selected_track_id"] = None
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None
if "matched_song" not in st.session_state:
    st.session_state["matched_song"] = None
if "song_input" not in st.session_state:
    st.session_state["song_input"] = ""

# ✅ Streamlit UI
st.title("🎵 Tamil Song Recommendation Engine (KNN)")
st.write("🔍 Start typing a song name and select from suggestions.")

# ✅ Step 1: Create Search Bar with Google-Style Auto-Suggestions
song_input = st.text_input("🎶 Enter Song Name:", value=st.session_state["song_input"])

# ✅ Step 2: Dynamically Filter Songs Matching the User Input
filtered_songs = df[df["name"].str.lower().str.contains(song_input.lower(), na=False)].sort_values(by="name")

# ✅ Step 3: Press "Enter" to Search for Recommendations
if song_input and st.session_state["song_input"] != song_input:
    st.session_state["song_input"] = song_input
    st.rerun()

# ✅ Step 4: Show Search Suggestions (Click to Autofill + Search)
if not filtered_songs.empty:
    suggested_songs = filtered_songs["name"].unique().tolist()[:5]  # Show top 5 matches
    st.markdown("#### 🔽 Suggestions:")
    for song in suggested_songs:
        if st.button(song, key=f"suggestion_{song}"):
            st.session_state["song_input"] = song  # Autofill the search bar
            st.rerun()  # 🔹 Instantly triggers search after clicking a suggestion

# ✅ Step 5: Button to Search (For Users Who Prefer Clicking)
if st.button("Get Recommendations"):
    matched_song, recommendations, error_message = recommend_songs(st.session_state["song_input"])

    if error_message:
        st.error(error_message)
    else:
        st.success(f"🎶 Showing recommendations for: **{st.session_state['song_input']}**")
        st.session_state["matched_song"] = matched_song
        st.session_state["recommendations"] = recommendations

# ✅ Display the Exact Matched Song First
if st.session_state["recommendations"] is not None:
    st.markdown("### 🎶 Recommended Songs")

    for index, row in st.session_state["recommendations"].iterrows():  # ✅ Correctly define `index`
        album_art_url = row.get("album_art_url", "https://via.placeholder.com/150")
        track_id = row["id"]

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.image(album_art_url, width=120)

        with col2:
            st.write(f"**{row['name']}**")
            st.write(f"*{row['artist']}*")

        with col3:
            unique_key = f"play_{row['id']}_{index}_{uuid.uuid4().hex}"  # ✅ Fully unique key
            if st.button("▶ Play", key=unique_key):
                st.session_state["selected_track_id"] = track_id
                st.rerun()

    st.markdown("---")

# ✅ Display the Recommendations After the Exact Match
if st.session_state["recommendations"] is not None:
    st.markdown("### 🎶 Recommended Songs")
    
    for _, row in st.session_state["recommendations"].iterrows():
        album_art_url = row.get("album_art_url", "https://via.placeholder.com/150")
        track_id = row["id"]  

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:  
            st.image(album_art_url, width=120)

        with col2:  
            st.write(f"**{row['name']}**")
            st.write(f"*{row['artist']}*")

        with col3:
            for index, row in st.session_state["recommendations"].iterrows():  # ✅ Define `index`
                unique_key = f"play_{row['id']}_{index}_{uuid.uuid4().hex}"
            unique_key = f"play_{row['id']}_{index}_{uuid.uuid4().hex}" # ✅ Add `index` to make keys unique
            if st.button("▶ Play", key=unique_key):
                st.session_state["selected_track_id"] = track_id
                st.rerun()

    st.markdown("---")

# ✅ Sticky Footer - Full Spotify Player
if st.session_state["selected_track_id"]:
    track_id = st.session_state["selected_track_id"]
    spotify_url = f"https://open.spotify.com/embed/track/{track_id}?autoplay=1"

    # ✅ Add Sticky Footer Player with CSS
    st.markdown(
        f"""
        <style>
        .sticky-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: black;
            padding: 10px;
            text-align: center;
            z-index: 100;
            box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.2);
        }}
        </style>
        <div class="sticky-footer">
            <iframe id="spotify-player" style="border-radius:12px" 
            src="{spotify_url}" width="100%" height="100px" frameborder="0" 
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
            loading="lazy"></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )
