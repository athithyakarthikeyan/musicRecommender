import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import time

# 🔹 Spotify API Credentials (Replace with your actual credentials)
CLIENT_ID = "39322097689b4647838e10649a39658d"         ## MY OWN SPOTIFY CLIENT ID
CLIENT_SECRET = "869557279be94ffd89c41a09c63ccc6a"     ## MY OWN SPOTIFY CLIENT SECRET

REDIRECT_URI = "http://localhost:8888/callback"  # Required for OAuth

# 🔹 Define OAuth scope (needed for accessing user playlists)
SCOPE = "playlist-read-private"

# 🔹 Authenticate with OAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID, 
                                               client_secret=CLIENT_SECRET, 
                                               redirect_uri=REDIRECT_URI, 
                                               scope=SCOPE))

# ✅ Fetch Track Metadata
def get_track_metadata(track):
    try:
        artist_id = track["artists"][0]["id"]
        artist_info = sp.artist(artist_id)
        genres = artist_info["genres"]

        # Fetch album image URL
        album_images = track["album"]["images"]
        album_art_url = album_images[0]["url"] if album_images else "https://via.placeholder.com/150"  # Default image if missing

        return {
            "id": track["id"],
            "name": track["name"],
            "artist": track["artists"][0]["name"],
            "album": track["album"]["name"],
            "release_date": track["album"]["release_date"],
            "popularity": track["popularity"],
            "duration_ms": track["duration_ms"],
            "genres": ", ".join(genres),
            "album_art_url": album_art_url  # ✅ Store album art URL
        }
    except Exception as e:
        print(f"❌ Error fetching metadata for track {track['name']}: {e}")
        return None

# ✅ Fetch Tracks from a Playlist
def get_user_playlist_tracks(playlist_id):
    try:
        results = sp.playlist_tracks(playlist_id)
        tracks = results["items"]
        
        tamil_track_list = []
        for item in tracks:
            track = item["track"]
            metadata = get_track_metadata(track)
            if metadata:
                tamil_track_list.append(metadata)
            time.sleep(0.2)  # To avoid rate limiting

        return pd.DataFrame(tamil_track_list)
    
    except Exception as e:
        print(f"❌ Error fetching playlist tracks: {e}")
        return None

# ✅ Main Execution
if __name__ == "__main__":
    user_playlists = sp.current_user_playlists()
    print("\n✅ Your Playlists:")
    for playlist in user_playlists["items"]:
        print(f"- {playlist['name']} (ID: {playlist['id']})")

    playlist_id = input("Enter Playlist ID: ").strip()
    df_tamil_tracks = get_user_playlist_tracks(playlist_id)

    if df_tamil_tracks is not None and not df_tamil_tracks.empty:
        print("\n✅ Tamil Songs Fetched Successfully!")
        df_tamil_tracks.to_csv("spotify_playlist_data.csv", index=False)  # ✅ Save to the correct CSV file
        print("\n✅ Data saved to `tamil_spotify_tracks.csv`!")
    else:
        print("\n❌ No Tamil songs found in the selected playlist.")
