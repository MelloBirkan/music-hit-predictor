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
        self.results = {}
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Fazer predições
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calcular métricas
            self.results[name] = {
                'auc_roc': roc_auc_score(y_test, y_prob),
                'precision': precision_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Salvar modelo
            model_path = Path('models/trained_models')
            model_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, f'models/trained_models/{name}.joblib')
            
            # Salvar o scaler junto com os modelos
            joblib.dump(preprocessor.scaler, 'models/trained_models/scaler.joblib')
            
    def plot_results(self):
        metrics = ['auc_roc', 'precision', 'f1']
        fig = go.Figure()
        
        for metric in metrics:
            values = [self.results[model][metric] for model in self.models.keys()]
            fig.add_trace(go.Bar(
                name=metric,
                x=list(self.models.keys()),
                y=values
            ))
        
        fig.update_layout(
            title='Comparação de Performance dos Modelos',
            barmode='group',
            xaxis_title='Modelo',
            yaxis_title='Pontuação'
        )
        
        return fig

def main():
    from data_preprocessing import DataPreprocessor
    
    # Carregar e preparar dados
    df = pd.read_csv('data/processed_data.csv')
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    
    # Treinar e avaliar modelos
    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Salvar resultados
    results_df = pd.DataFrame(trainer.results).round(3)
    results_df.to_csv('models/model_results.csv')

if __name__ == "__main__":
    main() 