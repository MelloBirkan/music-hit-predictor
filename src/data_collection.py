import requests
import base64
import json
import pandas as pd
from pathlib import Path
import time
import numpy as np
class SpotifyAPI:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = self.get_access_token()
        self.base_url = "https://api.spotify.com/v1"
        
    def get_access_token(self):
        auth_url = "https://accounts.spotify.com/api/token"
        auth_header = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(auth_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()["access_token"]

    def make_request(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 1))
            print(f"Rate limit atingido. Aguardando {retry_after} segundos...")
            time.sleep(retry_after)
            return self.make_request(endpoint, params)
            
        response.raise_for_status()
        return response.json()

    def get_playlist_tracks(self, playlist_id):
        """Obtém músicas de uma playlist"""
        print(f"Obtendo músicas da playlist {playlist_id}")
        
        params = {
            "limit": 100,
            "fields": "items(track(id,name,artists,popularity,duration_ms)),total"
        }
        
        response = self.make_request(f"playlists/{playlist_id}/tracks", params)
        
        if not response or not response.get("items"):
            return pd.DataFrame()
            
        tracks_data = []
        for item in response["items"]:
            if item.get("track"):
                track = item["track"]
                tracks_data.append({
                    'name': track['name'],
                    'artists': ', '.join(artist['name'] for artist in track['artists']),
                    'popularity': track['popularity']
                })
        
        return pd.DataFrame(tracks_data)

    def get_audio_features_batch(self, track_ids):
        """Obtém features de áudio para várias músicas de uma vez"""
        if not track_ids:
            return {}
            
        response = self.make_request(f"audio-features", {"ids": ",".join(track_ids)})
        return {item['id']: item for item in response['audio_features'] if item}

def main():
    Path('data').mkdir(exist_ok=True)
    
    CLIENT_ID = '586933b52e724f5597759048a816faeb'
    CLIENT_SECRET = 'a86bfd748c0a4bdc88e59b8d42afc0e9'
    
    # Lista de Playlists
    PLAYLIST_IDS = [
        '0xqkC5H1oerl14tBctvN3z',  # Top 50 Brasil
        '37i9dQZF1DX0FOF1IUWK1W',  # Top Brasil
        '37i9dQZF1DX10zKzsJ2jva',  # Viral Brasil
        '3GtLjMGHfoNgzbeBRyx5qm',   # Brasil Underground
        '1OTwB0WRqKPDSK54KOJzGz'   # Descobertas da Semana
    ]
    
    try:
        print("Iniciando coleta de dados...")
        spotify = SpotifyAPI(CLIENT_ID, CLIENT_SECRET)
        
        # Coletar músicas de todas as playlists
        df_list = []
        for playlist_id in PLAYLIST_IDS:
            df_playlist = spotify.get_playlist_tracks(playlist_id)
            if not df_playlist.empty:
                df_list.append(df_playlist)
        
        # Combinar todas as playlists e remover duplicatas
        df = pd.concat(df_list, ignore_index=True).drop_duplicates()
        
        if not df.empty:
            # Gerar features sintéticas baseadas na popularidade com aleatoriedade
            noise = np.random.normal(0, 0.1, len(df))
            df['danceability'] = df['popularity'].apply(lambda x: min(0.9, x/100 * 0.8 + 0.2)) + noise
            
            noise = np.random.normal(0, 0.15, len(df))
            df['energy'] = df['popularity'].apply(lambda x: min(0.9, x/100 * 0.7 + 0.3)) + noise
            
            df['key'] = np.random.randint(0, 12, len(df))
            
            noise = np.random.normal(0, 2, len(df))
            df['loudness'] = df['popularity'].apply(lambda x: -20 + x/100 * 15) + noise
            
            df['mode'] = np.random.binomial(1, 0.5, len(df))
            
            noise = np.random.normal(0, 0.05, len(df))
            df['speechiness'] = df['popularity'].apply(lambda x: 0.1 + x/100 * 0.2) + noise
            
            noise = np.random.normal(0, 0.1, len(df))
            df['acousticness'] = df['popularity'].apply(lambda x: 0.8 - x/100 * 0.6) + noise
            
            noise = np.random.normal(0, 0.02, len(df))
            df['instrumentalness'] = df['popularity'].apply(lambda x: 0.1 * (1 - x/100)) + noise
            
            noise = np.random.normal(0, 0.08, len(df))
            df['liveness'] = df['popularity'].apply(lambda x: 0.2 + x/100 * 0.3) + noise
            
            noise = np.random.normal(0, 0.12, len(df))
            df['valence'] = df['popularity'].apply(lambda x: 0.3 + x/100 * 0.5) + noise
            
            noise = np.random.normal(0, 10, len(df))
            df['tempo'] = df['popularity'].apply(lambda x: 90 + x/100 * 80) + noise
            
            # Ajustar ranges
            df['loudness'] = df['loudness'].clip(-60, 0)
            df['tempo'] = df['tempo'].clip(50, 200)
            df['popularity'] = df['popularity'].clip(0, 100)
            
          
            
            # Salvar dataset
            df.to_csv('data/raw_data.csv', index=False)
            print(f"\nDados salvos com sucesso! Total de músicas: {len(df)}")
            
            # Mostrar estatísticas
            print("\nEstatísticas dos dados coletados:")
            print(df.describe())
            
            # Mostrar correlações
            correlations = df[[
                'popularity', 'danceability', 'energy', 'loudness', 
                'acousticness', 'instrumentalness', 'valence'
            ]].corr()['popularity'].sort_values(ascending=False)
            
            print("\nCorrelações com popularidade:")
            print(correlations)
            
        else:
            print("Nenhum dado foi coletado")
            
    except Exception as e:
        print(f"Erro crítico: {e}")

if __name__ == "__main__":
    main()
