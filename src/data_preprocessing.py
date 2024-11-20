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
        popularity_threshold = df['popularity'].quantile(0.75)
        df['is_hit'] = (df['popularity'] >= popularity_threshold).astype(int)
        
        # Remove valores nulos e duplicatas
        df = df.dropna(subset=self.features)
        df = df.drop_duplicates(subset=self.features)
        
        # Adiciona ruído para evitar overfitting
        noise_factor = 0.05  # Aumentado para mais variabilidade
        for feature in self.features:
            noise = np.random.normal(0, noise_factor, size=len(df))
            df[feature] = df[feature] + noise
            # Garantir que os valores fiquem entre 0 e 1 para features normalizadas
            if feature not in ['key', 'loudness', 'tempo']:
                df[feature] = df[feature].clip(0, 1)
        
        # Normalização das features
        X = df[self.features].copy()
        df[self.features] = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.features,
            index=df.index
        )
        
        # Salvar o scaler
        Path('models/trained_models').mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, 'models/trained_models/scaler.joblib')
        
        return df

    def split_data(self, df):
        # Garantir que X mantenha os nomes das colunas
        X = df[self.features].copy()  # Usando .copy() para manter os nomes
        y = df['is_hit']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test

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