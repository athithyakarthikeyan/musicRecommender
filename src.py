import pandas as pd
import numpy as np
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# Set page config first before any other Streamlit commands
st.set_page_config(page_title="Tamil Music Recommender", layout="wide")

# Load the CSV file
@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)

csv_file = "dataset.csv"
df = load_data(csv_file)

# Drop unnecessary columns and ensure numerical data is clean
features = ["Danceability", "Energy", "Valence", "Tempo", "Popularity", "Duration (ms)"]
df_features = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

# Pre-compute normalized features
@st.cache_data
def compute_normalized_features(features_df):
    scaler = StandardScaler()
    return scaler.fit_transform(features_df)  # Return numpy array directly (faster)

df_features_scaled = compute_normalized_features(df_features)

# Weights for recommendation accuracy
weights = np.array([4.0, 3.5, 3.0, 0.5, 1.0, 0.3])
df_features_weighted = df_features_scaled * weights  # Element-wise multiplication

# Professional styling
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9 !important;
        }
        .stApp {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        h1 {
            color: #1DB954;
            font-weight: 600;
        }
        h2 {
            color: #191414;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Tamil Music Recommender")

# Song Selection (Allow clearing for blank state)
selected_song = st.selectbox("Select a song:", [""] + list(df["Track Name"].dropna().unique()), index=0)

# Ensure app starts blank
if not selected_song:
    st.warning("Please select a song to get recommendations.")
    st.stop()

# Get data of selected song
song_data = df[df["Track Name"] == selected_song].iloc[0]

# Display selected song details
st.markdown(f"## Now Playing: {song_data['Track Name']}")
st.write(f"**Artist:** {song_data['Artist Name(s)']}")
st.write(f"**Album:** {song_data['Album Name']}")

# Display album cover
if pd.notna(song_data.get("Album Cover URL", None)):
    st.image(song_data["Album Cover URL"], width=350)

# Embed Spotify Play Button
st.components.v1.html(f"""
    <iframe src="https://open.spotify.com/embed/track/{song_data['Track ID']}" 
    width="400" height="100" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
""", height=120)

# Improved recommendation algorithm
def recommend_songs(track_id, num_recommendations=5):
    selected_idx = df[df["Track ID"] == track_id].index
    if len(selected_idx) == 0:
        return pd.DataFrame()

    selected_idx = selected_idx[0]
    selected_features = df_features_weighted[selected_idx].reshape(1, -1)

    # Compute cosine similarity (vectorized for speed)
    similarities = 1 - cdist(selected_features, df_features_weighted, metric="cosine")[0]

    # Attach similarity scores directly to df
    df["Similarity"] = similarities

    # Filter out selected song & sort by similarity
    recommendations = df[df["Track ID"] != track_id].nlargest(20, "Similarity")  # Get top 20 similar songs

    # Randomize selection from top matches
    recommendations = recommendations.sample(n=min(len(recommendations), num_recommendations), random_state=np.random.randint(1000))

    return recommendations

# Get recommendations
recommendations = recommend_songs(song_data["Track ID"])

# Display recommendations
st.markdown("## Recommended Songs")
for _, row in recommendations.iterrows():
    col1, col2 = st.columns([1, 5])
    
    with col1:
        if pd.notna(row["Album Cover URL"]):
            st.image(row["Album Cover URL"], width=120)
    
    with col2:
        st.markdown(f"**{row['Track Name']}**")
        st.write(f"Artist: {row['Artist Name(s)']}")
        st.write(f"Album: {row['Album Name']}")

        # Embed Spotify Play Button
        st.components.v1.html(f"""
            <iframe src="https://open.spotify.com/embed/track/{row['Track ID']}" 
            width="350" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
        """, height=100)
