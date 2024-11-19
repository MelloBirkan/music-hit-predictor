import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo'
        ]

    def prepare_data(self, df):
        # Define hit baseado na popularidade (top 25%)
        popularity_threshold = df['popularity'].quantile(0.75)
        df['is_hit'] = (df['popularity'] >= popularity_threshold).astype(int)
        
        # Remove valores nulos
        df = df.dropna(subset=self.features + ['is_hit'])
        
        # Normalização das features
        df[self.features] = self.scaler.fit_transform(df[self.features])
        
        # Criar diretório se não existir
        model_path = Path('models/trained_models')
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Salvar o scaler
        joblib.dump(self.scaler, 'models/trained_models/scaler.joblib')
        
        return df

    def split_data(self, df):
        X = df[self.features]
        y = df['is_hit']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Carregar dados
    df = pd.read_csv('data/raw_data.csv')
    
    # Pré-processar
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_data(df)
    
    # Salvar dados processados
    df_processed.to_csv('data/processed_data.csv', index=False)
    
    print("Dados processados e scaler salvos com sucesso!")

if __name__ == "__main__":
    main() 