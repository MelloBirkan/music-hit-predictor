import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

class MusicDataGenerator:
    def __init__(self, n_samples=2500):
        self.n_samples = n_samples
        
    def generate_artist_names(self):
        first_names = ['The', 'DJ', 'Lil', 'Young', 'MC', 'Big', 'King', 'Queen', 'Lady', 'Sir']
        second_names = ['Dragon', 'Star', 'Beat', 'Flow', 'Rhythm', 'Wave', 'Sound', 'Voice', 'Moon', 'Sun']
        
        artists = []
        for _ in range(self.n_samples):
            if np.random.random() < 0.3:  # 30% chance de ter "The" no início
                artist = f"The {np.random.choice(second_names)}"
            else:
                artist = f"{np.random.choice(first_names)} {np.random.choice(second_names)}"
            artists.append(artist)
        return artists

    def generate_song_names(self):
        adjectives = ['Beautiful', 'Dark', 'Sweet', 'Crazy', 'Lost', 'Wild', 'Endless', 'Perfect', 'Blue', 'Golden']
        nouns = ['Love', 'Heart', 'Night', 'Dance', 'Dream', 'Life', 'Summer', 'Rain', 'Sky', 'Soul']
        
        songs = []
        for _ in range(self.n_samples):
            if np.random.random() < 0.5:
                song = f"{np.random.choice(adjectives)} {np.random.choice(nouns)}"
            else:
                song = np.random.choice(nouns)
            songs.append(song)
        return songs

    def generate_features(self):
        # Gerando características com distribuições realistas
        data = {
            'name': self.generate_song_names(),
            'artists': self.generate_artist_names(),
            'danceability': np.random.beta(7, 3, self.n_samples),  # Tendência para valores mais altos
            'energy': np.random.beta(5, 3, self.n_samples),
            'key': np.random.randint(0, 12, self.n_samples),
            'loudness': np.random.normal(-8, 4, self.n_samples),  # Média em -8 dB
            'mode': np.random.binomial(1, 0.6, self.n_samples),  # 60% maior
            'speechiness': np.random.beta(2, 15, self.n_samples),  # Geralmente baixo
            'acousticness': np.random.beta(3, 4, self.n_samples),
            'instrumentalness': np.random.beta(1, 8, self.n_samples),  # Geralmente muito baixo
            'liveness': np.random.beta(2, 8, self.n_samples),  # Geralmente baixo
            'valence': np.random.beta(5, 5, self.n_samples),  # Distribuição mais uniforme
            'tempo': np.random.normal(120, 20, self.n_samples),  # Média em 120 BPM
        }

        # Gerando popularidade com base nas características
        popularity = (
            data['danceability'] * 30 +
            data['energy'] * 25 +
            np.abs(data['loudness']) * 0.5 +
            (1 - data['instrumentalness']) * 15 +  # Músicas menos instrumentais tendem a ser mais populares
            data['valence'] * 15
        )
        
        # Normalizando popularidade para 0-100 e adicionando ruído
        popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min()) * 100
        popularity = popularity + np.random.normal(0, 5, self.n_samples)
        data['popularity'] = np.clip(popularity, 0, 100)

        # Criando DataFrame
        df = pd.DataFrame(data)
        
        # Ajustando os valores para ranges realistas
        df['loudness'] = np.clip(df['loudness'], -60, 0)
        df['tempo'] = np.clip(df['tempo'], 50, 200)
        
        return df

def main():
    # Criar diretório data se não existir
    os.makedirs('data', exist_ok=True)
    
    # Gerar dados
    generator = MusicDataGenerator(n_samples=2500)
    df = generator.generate_features()
    
    # Adicionar algumas correlações realistas
    # Músicas mais dançantes tendem a ter mais energia
    noise = np.random.normal(0, 0.1, len(df))
    df['energy'] = (df['danceability'] * 0.7 + noise).clip(0, 1)
    
    # Músicas acústicas tendem a ter menos energia
    df['energy'] = (df['energy'] * (1 - df['acousticness'] * 0.5)).clip(0, 1)
    
    # Salvar dados
    df.to_csv('data/raw_data.csv', index=False)
    print(f"Dados gerados e salvos em 'data/raw_data.csv'")
    
    # Mostrar algumas estatísticas
    print("\nEstatísticas dos dados gerados:")
    print(df.describe())
    
    # Mostrar correlações interessantes
    correlations = df[[
        'popularity', 'danceability', 'energy', 'loudness', 
        'acousticness', 'instrumentalness', 'valence'
    ]].corr()['popularity'].sort_values(ascending=False)
    
    print("\nCorrelações com popularidade:")
    print(correlations)

if __name__ == "__main__":
    main() 