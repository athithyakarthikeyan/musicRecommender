import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import time

# 🎵 Spotify API Credentials
CLIENT_ID = "39322097689b4647838e10649a39658d"
CLIENT_SECRET = "869557279be94ffd89c41a09c63ccc6a"
REDIRECT_URI = "http://localhost:8888/callback"
SCOPE = "playlist-read-private"

# 🔹 Authenticate with Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID, 
                                               client_secret=CLIENT_SECRET, 
                                               redirect_uri=REDIRECT_URI, 
                                               scope=SCOPE))

# ✅ Fetch Track Metadata
def get_track_metadata(track):
    """Extract metadata from a Spotify track"""
    try:
        artist_id = track["artists"][0]["id"]
        artist_info = sp.artist(artist_id)
        genres = artist_info["genres"]

        album_images = track["album"]["images"]
        album_art_url = album_images[0]["url"] if album_images else "https://via.placeholder.com/150"

        return {
            "id": track["id"],
            "name": str(track["name"]),  # ✅ Ensure it's a string
            "artist": track["artists"][0]["name"],
            "album": track["album"]["name"],
            "release_date": track["album"]["release_date"],
            "popularity": track["popularity"],
            "duration_ms": track["duration_ms"],
            "genres": ", ".join(genres),
            "album_art_url": album_art_url
        }
    except Exception as e:
        print(f"❌ Error fetching metadata for track: {e}")
        return None

# ✅ Fetch All Tracks from a Playlist
def get_playlist_tracks(playlist_id):
    """Fetch all tracks from the given Spotify playlist"""
    
    offset = 0
    limit = 100  
    all_tracks = []

    while True:
        results = sp.playlist_tracks(playlist_id, offset=offset, limit=limit)
        tracks = results["items"]
        
        if not tracks:
            break  

        for item in tracks:
            track = item["track"]
            metadata = get_track_metadata(track)
            if metadata:
                all_tracks.append(metadata)
            time.sleep(0.2)  

        offset += limit  

    df = pd.DataFrame(all_tracks)
    df["name"] = df["name"].fillna("").astype(str)  # ✅ Ensure names are always strings
    df.to_csv("spotify_playlist_data.csv", index=False)
    print(f"\n✅ Fetched {len(all_tracks)} songs and saved to `spotify_playlist_data.csv`!")
    return df

# ✅ Run the script
if __name__ == "__main__":
    playlist_id = "5lfuu0un8XjAtUdxwtqjm4"
    print("\n🎵 Fetching all songs from Spotify playlist...")
    df = get_playlist_tracks(playlist_id)
