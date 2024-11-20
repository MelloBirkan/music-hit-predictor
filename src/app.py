import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from data_preprocessing import DataPreprocessor

class HitPredictorApp:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.load_models()
        self.load_scaler()
        
    def load_scaler(self):
        try:
            self.scaler = joblib.load('models/trained_models/scaler.joblib')
        except:
            st.error("Erro: Scaler não encontrado. Execute o treinamento primeiro.")
            st.stop()
    
    def load_models(self):
        try:
            self.models = {}
            model_path = Path('models/trained_models')
            for model_file in model_path.glob('*.joblib'):
                if 'scaler' not in str(model_file):
                    model_name = model_file.stem
                    self.models[model_name] = joblib.load(model_file)
        except:
            st.error("Erro: Modelos não encontrados. Execute o treinamento primeiro.")
            st.stop()
    
    def predict_hit(self, features_df, model_name):
        """
        Recebe um DataFrame com features e retorna a probabilidade
        """
        model = self.models[model_name]
        prediction = model.predict_proba(features_df)[0]
        return prediction[1]
    
    def run(self):
        st.title('🎵 Hit Predictor - Previsão de Sucessos Musicais')
        
        tab1, tab2 = st.tabs(['Fazer Previsão', 'Comparar Modelos'])
        
        with tab1:
            st.header('Faça sua previsão')
            
            # Inputs para características da música
            col1, col2 = st.columns(2)
            
            with col1:
                danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
                energy = st.slider('Energy', 0.0, 1.0, 0.5)
                key = st.slider('Key', 0, 11, 5)
                loudness = st.slider('Loudness', -60.0, 0.0, -30.0)
                mode = st.selectbox('Mode', [0, 1])
                
            with col2:
                speechiness = st.slider('Speechiness', 0.0, 1.0, 0.5)
                acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5)
                instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.5)
                liveness = st.slider('Liveness', 0.0, 1.0, 0.5)
                valence = st.slider('Valence', 0.0, 1.0, 0.5)
                tempo = st.slider('Tempo', 0.0, 250.0, 120.0)
            
            # Criar DataFrame com features
            features_df = pd.DataFrame(
                [[
                    danceability, energy, key, loudness, mode,
                    speechiness, acousticness, instrumentalness,
                    liveness, valence, tempo
                ]], 
                columns=self.preprocessor.features
            )
            
            model_name = st.selectbox('Escolha o modelo', list(self.models.keys()))
            
            if st.button('Prever'):
                # Normalizar mantendo o DataFrame
                features_normalized = pd.DataFrame(
                    self.scaler.transform(features_df),
                    columns=self.preprocessor.features
                )
                
                # Fazer previsão
                hit_probability = self.predict_hit(features_normalized, model_name)
                
                # Mostrar resultado
                st.header(f'Probabilidade de ser um Hit: {hit_probability:.1%}')
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = hit_probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilidade de Sucesso"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "darkgray"}
                        ]
                    }
                ))
                st.plotly_chart(fig)
        
        with tab2:
            st.header('Comparação de Modelos')
            
            try:
                # Carregar resultados
                results_df = pd.read_csv('models/model_results.csv')
                st.dataframe(results_df)
                
                # Plotar gráfico de comparação
                fig = go.Figure()
                
                for metric in ['auc_roc', 'precision', 'f1']:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=results_df['model'],
                        y=results_df[metric]
                    ))
                
                fig.update_layout(
                    title='Comparação de Performance dos Modelos',
                    barmode='group',
                    xaxis_title='Modelo',
                    yaxis_title='Pontuação'
                )
                st.plotly_chart(fig)
            
            except Exception as e:
                st.error(f"Erro ao carregar resultados: {str(e)}")
                st.error("Execute o treinamento dos modelos primeiro.")

if __name__ == "__main__":
    app = HitPredictorApp()
    app.run()
