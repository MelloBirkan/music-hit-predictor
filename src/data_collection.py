import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

class SpotifyDataCollector:
    def __init__(self, client_id, client_secret):
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
        )

    def get_track_features(self, track_id):
        try:
            # Obtém características da música
            features = self.sp.audio_features(track_id)[0]
            # Obtém metadados da música
            track_info = self.sp.track(track_id)
            
            features['popularity'] = track_info['popularity']
            features['name'] = track_info['name']
            features['artists'] = ', '.join([artist['name'] for artist in track_info['artists']])
            
            return features
        except:
            return None

    def collect_playlist_tracks(self, playlist_id):
        tracks = []
        results = self.sp.playlist_tracks(playlist_id)
        
        while results:
            for item in results['items']:
                if item['track']:
                    track_id = item['track']['id']
                    features = self.get_track_features(track_id)
                    if features:
                        tracks.append(features)
            
            if len(tracks) >= 2500:  # Limite de músicas
                break
                
            results = self.sp.next(results) if results['next'] else None
            time.sleep(0.5)  # Evitar limite de requisições
            
        return pd.DataFrame(tracks)

def main():
    # Substitua com suas credenciais do Spotify
    CLIENT_ID = '586933b52e724f5597759048a816faeb'
    CLIENT_SECRET = 'a86bfd748c0a4bdc88e59b8d42afc0e9'
    
    collector = SpotifyDataCollector(CLIENT_ID, CLIENT_SECRET)
    
    # Lista de playlists populares para coletar dados
    playlists = [
        'spotify:playlist:37i9dQZF1DXcBWIGoYBM5M',  # Today's Top Hits
        'spotify:playlist:37i9dQZF1DX0XUsuxWHRQd'   # RapCaviar
    ]
    
    all_tracks = pd.DataFrame()
    for playlist_id in playlists:
        df = collector.collect_playlist_tracks(playlist_id)
        all_tracks = pd.concat([all_tracks, df], ignore_index=True)
    
    # Salvar dados
    all_tracks.to_csv('data/raw_data.csv', index=False)

if __name__ == "__main__":
    main() 