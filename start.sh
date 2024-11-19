#!/bin/bash

# Executar coleta de dados
python src/data_collection.py

# Executar pré-processamento
python src/data_preprocessing.py

# Executar treinamento do modelo
python src/model_training.py

# Iniciar a aplicação Streamlit
streamlit run src/app.py 