# Imagem base
FROM python:3.9.16-slim

# Diretório de trabalho
WORKDIR /app

# Copiar arquivos necessários
COPY requirements.txt .
COPY src/ ./src/
COPY start.sh .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Criar diretórios necessários
RUN mkdir -p data models/trained_models

# Tornar o script executável
RUN chmod +x start.sh

# Expor porta para o Streamlit
EXPOSE 8501

# Comando para executar a aplicação
CMD ["./start.sh"]