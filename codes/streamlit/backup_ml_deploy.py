# Deploy de Aplicações Preditivas com Streamlit

# Imports
import time
import numpy as np
import pandas as pd
import streamlit as st
import sklearn.metrics
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class SessionState(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

##### Programando a Barra Superior da Aplicação Web #####

# Títulos Estilizados
st.markdown("# Deploy de Modelos de Machine Learning")
st.markdown("Murilo M Silvestrini")
st.title("Modelos")

##### Programando a Barra Lateral de Navegação da Aplicação Web #####

# Cabeçalho lateral
st.sidebar.header('Dataset e Hiperparâmetros')
st.sidebar.markdown("""**Selecione o Dataset Desejado**""")
Dataset = st.sidebar.selectbox('Dataset',('Iris', 'Wine', 'Breast Cancer'))
Split = st.sidebar.slider('Escolha o Percentual de Divisão dos Dados em Treino e Teste (padrão = 70/30):', 0.1, 0.9, 0.70)
st.sidebar.markdown("""**Selecione os Hiperparâmetros Para o Modelo de Regressão Logística**""")

# Dicionário para mapear nomes técnicos e amigáveis
solvers_dict = {
    "BFGS (Padrão)": "lbfgs",
    "Newton-CG": "newton-cg",
    "Liblinear (Para conjuntos pequenos)": "liblinear",
    "Método do Gradiente Estocástico": "sag"
}

# Use o nome amigável na barra lateral
friendly_solvers = list(solvers_dict.keys())
Solver = st.sidebar.selectbox('Algoritmo', friendly_solvers)

# Mapeie de volta para o nome técnico ao definir os parâmetros
technical_solver = solvers_dict[Solver]

Penality = st.sidebar.radio("Regularização:", ('none', 'l1', 'l2', 'elasticnet'))
Tol = st.sidebar.text_input("Tolerância Para Critério de Parada (default = 1e-4):", "1e-4")
Max_Iteration = st.sidebar.text_input("Número de Iterações (default = 50):", "50")

# Validação dos hiperparâmetros
try:
    tol_value = float(Tol)
except ValueError:
    st.sidebar.warning("Por favor, insira um valor válido para a Tolerância.")
    tol_value = 1e-4

try:
    max_iter_value = int(Max_Iteration)
except ValueError:
    st.sidebar.warning("Por favor, insira um valor válido para Número de Iterações.")
    max_iter_value = 50

# Atualização do dicionário de parâmetros
parameters = {'Penality': Penality, 'Tol': tol_value, 'Max_Iteration': max_iter_value, 'Solver': technical_solver}


##### Funções Para Carregar e Preparar os Dados #####

# Função para carregar o dataset
def carrega_dataset(dataset):
    if dataset == 'Iris':
        dados = sklearn.datasets.load_iris()
    elif dataset == 'Wine':
         dados = sklearn.datasets.load_wine()
    elif dataset == 'Breast Cancer':
         dados = sklearn.datasets.load_breast_cancer()
    
    return dados

# Função para preparar os dados e fazer a divisão em treino e teste
def prepara_dados(dados, split):
    X_treino, X_teste, y_treino, y_teste = train_test_split(dados.data, dados.target, test_size = float(split), random_state = 42)
    scaler = MinMaxScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)
    return (X_treino, X_teste, y_treino, y_teste)

##### Função Para o Modelo de Machine Learning #####  

# Função para o modelo
def cria_modelo(parameters):
    X_treino, X_teste, y_treino, y_teste = prepara_dados(Data, Split) 
    clf = LogisticRegression(penalty = parameters['Penality'], 
                             solver = parameters['Solver'], 
                             max_iter = int(parameters['Max_Iteration']), 
                             tol = float(parameters['Tol']))
    clf = clf.fit(X_treino, y_treino)
    prediction = clf.predict(X_teste)
    accuracy = sklearn.metrics.accuracy_score(y_teste, prediction)
    precision = sklearn.metrics.precision_score(y_teste, prediction, average='weighted')
    recall = sklearn.metrics.recall_score(y_teste, prediction, average='weighted')
    f1 = sklearn.metrics.f1_score(y_teste, prediction, average='weighted')
    cm = confusion_matrix(y_teste, prediction)
    dict_value = {"modelo":clf, "acuracia": accuracy, "precision": precision, "recall": recall, "f1": f1, "previsao":prediction, "y_real": y_teste, "Metricas":cm, "X_teste": X_teste}
    return(dict_value)



##### Programando o Corpo da Aplicação Web ##### 

st.markdown("""Resumo dos Dados""")
st.write("Nome do Dataset:", Dataset)
Data = carrega_dataset(Dataset)
targets = Data.target_names
Dataframe = pd.DataFrame (Data.data, columns = Data.feature_names)
Dataframe['target'] = pd.Series(Data.target)
Dataframe['target labels'] = pd.Series(targets[i] for i in Data.target)
st.write("Visão Geral dos Atributos:")
st.write(Dataframe)

##### Programando o Botão de Ação ##### 

# Criação ou obtenção do estado de sessão
session_state = st.session_state if hasattr(st, 'session_state') else SessionState()

# Verifica se o modelo já foi treinado
if not hasattr(session_state, 'modelo'):
    session_state.modelo = None

if st.sidebar.button("Treinar o Modelo"):
    with st.spinner('Carregando o Dataset...'):
        time.sleep(.5)
    st.success("Dataset Carregado!")
    session_state.modelo = cria_modelo(parameters)
    modelo = cria_modelo(parameters) 
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    with st.spinner('Treinando o Modelo...'):
        time.sleep(1)
    st.success("Modelo Treinado") 
    labels_reais = [targets[i] for i in modelo["y_real"]]
    labels_previstos = [targets[i] for i in modelo["previsao"]]
    st.subheader("Previsões do Modelo nos Dados de Teste")
    st.write(pd.DataFrame({"Valor Real" : modelo["y_real"], 
                           "Label Real" : labels_reais, 
                           "Valor Previsto" : modelo["previsao"], 
                           "Label Previsto" :  labels_previstos,}))
    matriz = modelo["Metricas"]
    st.subheader("Matriz de Confusão nos Dados de Teste")
    st.write(matriz)
    st.subheader("Métricas Adicionais")
    st.write(f"Precision: {modelo['precision']:.2f}")
    st.write(f"Recall: {modelo['recall']:.2f}")
    st.write(f"F1-Score: {modelo['f1']:.2f}")
    


    if session_state.modelo:  # Supondo que você ainda está usando session_state para armazenar o modelo

        # Gerando dados aleatórios para previsão
        user_input = []
        st.subheader("Dados de Simulação para Nova Previsão")
        for feature_name in Data.feature_names:
            # Aqui estou assumindo que você quer valores aleatórios entre 0 e 1, ajuste conforme necessário
            value = np.random.rand()
            st.write(f"{feature_name}: {value:.4f}")
            user_input.append(value)

        # Fazendo a previsão com os dados aleatórios
        prediction = session_state.modelo["modelo"].predict([user_input])[0]
        st.write(f"Previsão com Dados Simulados: {targets[prediction]}")
        
        st.write("""
        Em uma aplicação em produção, em vez de usar valores aleatórios como fizemos aqui, 
        o usuário teria a opção de inserir seus próprios dados e obter previsões em tempo real.
        """)


    # Informação adicional sobre a previsão real
    st.markdown("""
    #### Informação Adicional:

    Na prática, para usar o modelo treinado para fazer previsões em um ambiente de produção, você seguiria os seguintes passos:

    1. **Treine o Modelo:** Uma vez que o modelo é treinado, você irá serializar o modelo, ou seja, converter o modelo treinado em um formato que pode ser salvo em disco.
    2. **Salvar o Modelo:** O modelo serializado é então salvo em disco.
    3. **Integração com o Sistema de Produção:** Ao receber novos dados para previsão, o sistema de produção irá desserializar o modelo, carregá-lo e usar o modelo carregado para fazer previsões.
    4. **Previsão:** O modelo carregado é usado para fazer previsões que são então retornadas ao usuário ou outro sistema.

    Esta abordagem garante que o modelo treinado possa ser reutilizado sem a necessidade de retré-lo toda vez que precisarmos fazer uma previsão. Em uma implementação real, isso é crítico para garantir a eficiência e a escalabilidade do sistema.

    A razão pela qual não podemos seguir exatamente este fluxo aqui é devido às limitações de manter o estado entre as execuções na plataforma Streamlit. Em um ambiente de produção completo, tal limitação não existiria.
    """)

