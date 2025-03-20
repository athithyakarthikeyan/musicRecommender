import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Replace with your Spotify App credentials
CLIENT_ID = "39322097689b4647838e10649a39658d"         ## MY OWN SPOTIFY CLIENT ID
CLIENT_SECRET = "869557279be94ffd89c41a09c63ccc6a"     ## MY OWN SPOTIFY CLIENT SECRET

REDIRECT_URI = "http://localhost:8888/callback"  # Required for OAuth

# Define OAuth scope to access user playlists
SCOPE = "playlist-read-private"

# Authenticate with OAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID, 
                                               client_secret=CLIENT_SECRET, 
                                               redirect_uri=REDIRECT_URI, 
                                               scope=SCOPE))
print("✅ OAuth Authentication Successful!")
