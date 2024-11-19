import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, f1_score, confusion_matrix
import joblib
import plotly.graph_objects as go
from pathlib import Path

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42)
        }
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Lista para armazenar resultados
        results_list = []
        
        for name, model in self.models.items():
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Fazer predições
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calcular métricas
            results = {
                'model': name,
                'auc_roc': roc_auc_score(y_test, y_prob),
                'precision': precision_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            results_list.append(results)
            
            # Salvar modelo
            model_path = Path('models/trained_models')
            model_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, f'models/trained_models/{name}.joblib')
        
        # Criar DataFrame com resultados
        results_df = pd.DataFrame(results_list)
        
        # Salvar resultados
        results_df.to_csv('models/model_results.csv', index=False)
        print("Modelos treinados e resultados salvos com sucesso!")
        print("\nResultados:")
        print(results_df)

def main():
    from data_preprocessing import DataPreprocessor
    
    # Carregar e preparar dados
    df = pd.read_csv('data/processed_data.csv')
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    
    # Treinar e avaliar modelos
    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main() 