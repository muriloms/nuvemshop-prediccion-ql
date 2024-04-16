# Deploy de Aplicações Preditivas com Streamlit

# Imports
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from joblib import load

class SessionState(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

# Carregar os transformadores e modelos
encoder_loaded = load('models/ordinal_encoder.joblib')
scaler_loaded = load('models/scaler.joblib')
loaded_features = load('models/selected_features.pkl')
best_rf_model = load('models/best_rf_model.pkl')
best_ada_model = load('models/best_ada_model.pkl')
best_svm_model = load('models/best_svm_model.pkl')

##### Programando a Barra Superior da Aplicação Web #####

img_logo = Image.open('./imgs/logo.png')
st.image(img_logo)

# Títulos Estilizados
st.markdown("## Previsão de Quality Leads (QL)")
st.markdown("Murilo M Silvestrini")
st.markdown("---")
st.markdown("**Selecione os dados de um cliente que completou o período de teste de 30 dias e clique no botão abaixo. O sistema realizará uma previsão sobre se o cliente decidirá continuar utilizando o serviço e se tornará um assinante pagante ou se optará por abandonar o serviço.**")
st.markdown("---")

##### Programando a Barra Lateral de Navegação da Aplicação Web #####

# Coleta de dados de entrada
st.sidebar.header('Selecione os dados para previsão')

# Variáveis categóricas
country = st.sidebar.selectbox('Country', ['BR', 'AR', 'MX', 'CO','CL'])
creation_platform = st.sidebar.selectbox('Creation Platform', ['desktop', 'mobile_web', 'mobile_app','tablet'])
source_pulido = st.sidebar.selectbox('Source Pulido', [
    "Facebook CPC",
    "Other",
    "Google CPC no Brand",
    "Brand",
    "Google Organic",
    "Google CPC DSA",
    "partners",
    "Direct",
    "Store Referral",
    "Google CPC Competitors"
])

# Variáveis numéricas
admin_visits = st.sidebar.number_input('Admin Visits', value=0.0, format="%.2f")
intercom_conversations = st.sidebar.number_input('Intercom Conversations', value=0.0, format="%.2f")
creation_weekday = st.sidebar.slider('Creation Weekday', 0, 6, 1)
creation_hour = st.sidebar.slider('Creation Hour', 0, 23, 12)
products_with_description = st.sidebar.number_input('Products with Description', value=0.0, format="%.2f")
total_products_with_images = st.sidebar.slider('Total Products with Images', 0, 100, 10)
total_product_categories = st.sidebar.number_input('Total Product Categories', value=0.0, format="%.2f")
total_events_on_Android = st.sidebar.number_input('Total Events on Android', value=0.0, format="%.2f")
total_events_on_Web = st.sidebar.number_input('Total Events on Web', value=0.0, format="%.2f")
total_events_on_iOS = st.sidebar.number_input('Total Events on iOS', value=0.0, format="%.2f")

# Continuação com a definição dos hiperparâmetros (como no código existente)
# Inclua aqui as partes existentes sobre a configuração de hiperparâmetros e modelos

# Criação de um DataFrame com os dados coletados
data = {
    'country': [country],
    'creation_platform': [creation_platform],
    'admin_visits': [admin_visits],
    'intercom_conversations': [intercom_conversations],
    'source_pulido': [source_pulido],
    'creation_weekday': [creation_weekday],
    'creation_hour': [creation_hour],
    'products_with_description': [products_with_description],
    'total_products_with_images': [total_products_with_images],
    'total_product_categories': [total_product_categories],
    'total_events_on_Android': [total_events_on_Android],
    'total_events_on_Web': [total_events_on_Web],
    'total_events_on_iOS': [total_events_on_iOS]
}

df_user_input = pd.DataFrame(data)

##### Funções Para Carregar e Preparar os Dados #####

# Checando quais colunas são do tipo 'object', indicando variáveis categóricas
categorical_columns = df_user_input.select_dtypes(include=['object']).columns

# Identificando colunas numéricas (int e float)
numeric_columns = df_user_input.select_dtypes(include=['int64', 'float64']).columns

# Aplicando o encoder e o scaler
df_user_input[categorical_columns] = encoder_loaded.transform(df_user_input[categorical_columns])
df_user_input[numeric_columns] = scaler_loaded.transform(df_user_input[numeric_columns])

# Filtrando as colunas com base nas características selecionadas
df_user_input_filtered = df_user_input[loaded_features]

# Aqui você poderia adicionar o código para fazer previsões usando os modelos carregados
predictions_rf = best_rf_model.predict(df_user_input_filtered)
predictions_ada = best_ada_model.predict(df_user_input_filtered)
predictions_svm = best_svm_model.predict(df_user_input_filtered)


##### Programando o Botão de Ação ##### 
# Botão para gerar resultados
if st.button('Gerar Resultados'):
    # Processar os dados de entrada (assumindo que isso já está implementado)
    df_user_input_filtered = df_user_input[loaded_features]

    if predictions_ada == 0:
        st.markdown("---")
        st.markdown("# Cliente vai optar por abandonar o serviço")
        st.markdown("---")
    else:
        st.markdown("---")
        st.markdown("# Cliente decidirá continuar utilizando o serviço")
        st.markdown("---")

    # Fazendo previsões e exibindo resultados
    if hasattr(best_rf_model, 'predict_proba'):
        predictions_rf = best_rf_model.predict(df_user_input_filtered)
        probabilities_rf = best_rf_model.predict_proba(df_user_input_filtered)
        st.write("Random Forest Predictions:", predictions_rf)
        st.write("Random Forest Probabilities:", probabilities_rf)
    
    if hasattr(best_ada_model, 'predict_proba'):
        predictions_ada = best_ada_model.predict(df_user_input_filtered)
        probabilities_ada = best_ada_model.predict_proba(df_user_input_filtered)
        st.write("AdaBoost Predictions:", predictions_ada)
        st.write("AdaBoost Probabilities:", probabilities_ada)

    if hasattr(best_svm_model, 'predict_proba'):
        predictions_svm = best_svm_model.predict(df_user_input_filtered)
        probabilities_svm = best_svm_model.predict_proba(df_user_input_filtered)
        st.write("SVM Predictions:", predictions_svm)
        st.write("SVM Probabilities:", probabilities_svm)
    else:
        predictions_svm = best_svm_model.predict(df_user_input_filtered)
        st.write("SVM Predictions:", predictions_svm)
        st.write("SVM does not support probability estimates")



