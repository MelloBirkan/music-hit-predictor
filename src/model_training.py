import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, f1_score, confusion_matrix
import joblib
import plotly.graph_objects as go
from pathlib import Path
from data_preprocessing import DataPreprocessor

class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=3,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.01,
                max_depth=2,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=0.01,
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        }
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        results_list = []
        
        for name, model in self.models.items():
            print(f"\nTreinando modelo: {name}")
            
            # Garantir que X_train e X_test mantenham os nomes das colunas
            X_train_df = pd.DataFrame(X_train, columns=self.preprocessor.features)
            X_test_df = pd.DataFrame(X_test, columns=self.preprocessor.features)
            
            # Treinar modelo
            model.fit(X_train_df, y_train)
            
            # Fazer predições
            y_pred = model.predict(X_test_df)
            y_prob = model.predict_proba(X_test_df)[:, 1]
            
            # Calcular métricas
            results = {
                'model': name,
                'auc_roc': roc_auc_score(y_test, y_prob),
                'precision': precision_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'feature_importance': None
            }
            
            # Adicionar importância das features para RF e GB
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = dict(zip(
                    self.preprocessor.features,
                    model.feature_importances_
                ))
            
            results_list.append(results)
            
            # Imprimir matriz de confusão
            print(f"\nMatriz de Confusão - {name}:")
            print(confusion_matrix(y_test, y_pred))
            
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
    # Carregar e preparar dados
    df = pd.read_csv('data/processed_data.csv')
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    
    # Treinar e avaliar modelos
    trainer = ModelTrainer(preprocessor)
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main() 