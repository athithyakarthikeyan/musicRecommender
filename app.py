#
# Algorithm: Cosine similarity on song metadata and Spotify API
#

import streamlit as st
import pandas as pd
import pickle
import difflib

# ✅ Load preprocessed data
df = pd.read_csv("spotify_playlist_data.csv")
with open("similarity_matrix.pkl", "rb") as f:
    similarity_matrix = pickle.load(f)

# ✅ Function to Find Closest Song Matches
def find_best_match(input_song):
    df["name"] = df["name"].fillna("").astype(str)  # ✅ Convert NaN to empty strings
    matches = difflib.get_close_matches(input_song.lower(), df["name"].str.lower(), n=5, cutoff=0.3)
    if matches:
        return df[df["name"].str.lower() == matches[0]]["name"].values[0]
    return None

# ✅ Recommendation Function
def recommend_songs(song_input, num_recommendations=20):
    matched_song = find_best_match(song_input)
    if matched_song is None:
        return None, f"❌ No song found matching '{song_input}'. Try again."

    song_idx = df[df["name"] == matched_song].index[0]
    similarity_scores = list(enumerate(similarity_matrix[song_idx]))
    sorted_songs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

    recommendations = df.iloc[[i[0] for i in sorted_songs]][["id", "name", "artist", "album", "album_art_url"]]
    return recommendations, None

# ✅ Initialize session state for recommendations and Spotify player
if "selected_track_id" not in st.session_state:
    st.session_state["selected_track_id"] = None
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None

# ✅ Callback function to update session state when a song is played
def play_song(track_id):
    st.session_state["selected_track_id"] = track_id

# ✅ Streamlit UI with Sticky Spotify Player
st.markdown(
    """
    <style>
    /* Sticky footer for the Spotify player */
    .sticky-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: black;
        padding: 10px;
        text-align: center;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎵 Tamil Song Recommendation Engine \n ")
st.header("Using Cosine Simailairty Alogrithm.")
st.write("🔍 Enter a song name (or part of a song name) to get recommendations.")

with st.form("song_search"):
    song_input = st.text_input("🎶 Enter Song Name:", "")
    submitted = st.form_submit_button("Get Recommendations")

if submitted and song_input:
    recommendations, error_message = recommend_songs(song_input)

    if error_message:
        st.error(error_message)
    else:
        st.success(f"🎶 Showing recommendations for: **{song_input}**")
        st.session_state["recommendations"] = recommendations  # ✅ Store recommendations in session state

# ✅ Display the Recommendations (Persist after clicking Play)
if st.session_state["recommendations"] is not None:
    for _, row in st.session_state["recommendations"].iterrows():
        album_art_url = row.get("album_art_url", "https://via.placeholder.com/150")  # Default Image
        track_id = row["id"]  # Spotify Track ID

        col1, col2, col3 = st.columns([1, 3, 1])  # Left: Image | Middle: Text | Right: Play Button
        with col1:  # ✅ Left Side - Album Art
            st.image(album_art_url, width=120)

        with col2:  # ✅ Middle - Song Name & Artist
            st.write(f"**{row['name']}**")
            st.write(f"*{row['artist']}*")

        with col3:  # ✅ Right - Play Button (HTML Symbol)
            unique_key = f"play_{row['id']}_{_}"  # ✅ Add loop index `_` to make the key unique
            if st.button("▶ Play", key=unique_key, on_click=play_song, args=(track_id,)):
                st.session_state["selected_track_id"] = track_id  # ✅ Update track ID
                st.rerun()  # ✅ Refresh UI to trigger autoplay

    st.markdown("---")  # Add a separator

# ✅ Sticky Footer - Full Spotify Player (Forcing Autoplay)
if st.session_state["selected_track_id"]:
    track_id = st.session_state["selected_track_id"]
    spotify_url = f"https://open.spotify.com/embed/track/{track_id}?autoplay=1"

    st.markdown(
        f"""
        <div class="sticky-footer">
            <iframe id="spotify-player" style="border-radius:12px" 
            src="{spotify_url}" width="100%" height="80px" frameborder="0" 
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
            loading="lazy"></iframe>

           
        </div>
        """,
        unsafe_allow_html=True,
    )
