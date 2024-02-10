# Bibliotecas e Pacotes

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import requests
import os
from google.cloud import bigquery
import pandas as pd
import configparser
import glob
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


# Classes
class API_Key_Etherscan:
    def __init__(self):
        self.api_key = None  # Atributo para armazenar a chave
 
    def carregar_chave(self, api):
        self.api_key = api

    def exibir_chave(self):
        if self.api_key is not None:
            # Simulação de exibição do DataFrame
            print(self.api_key)
        else:
            print("DataFrame está vazio.")
            
    def obter_chave(self):
        # Retorna a chave da API Etherscan
        if self.api_key is not None:
            return self.api_key
        else:
            adicionar_mensagem("API Key da Etherscan não foi carregada.")
            return None

class BigQueryClient:
    def __init__(self):
        self.client = None
        self.credentials_file = ""

    def selecionar_arquivo_credenciais(self):
        # Abre um diálogo para o usuário selecionar o arquivo de credenciais
        self.credentials_file = filedialog.askopenfilename(
            title="Selecione o arquivo de credenciais do Google BigQuery",
            filetypes=[("JSON files", "*.json")])
        return self.credentials_file

    def adicionar_mensagem(self, mensagem):
        print(mensagem)  

    def autenticar_bigquery(self):
        if self.selecionar_arquivo_credenciais() and os.path.isfile(self.credentials_file):
            try:
                self.client = bigquery.Client.from_service_account_json(self.credentials_file)
                adicionar_mensagem("==================== Etapa 1 ====================\n")
                adicionar_mensagem("Autenticação das credenciais do Google BigQuery foi bem-sucedida.")
            except Exception as e:
                self.adicionar_mensagem(f"Erro de autenticação: {str(e)}")
                self.client = None
        else:
            self.adicionar_mensagem("O arquivo de credenciais não foi selecionado ou não é válido.")

    def get_client(self):
        # Retorna o cliente autenticado, se disponível
        if self.client is not None:
            return self.client
        else:
            self.adicionar_mensagem("Cliente não está autenticado.")
            return None


class Comite_AM:
    def __init__(self):
        """Inicializa os resultados dos modelos com valores padrão ou nulos."""
        self.modelo1 = None
        self.modelo2 = None
        self.modelo3 = None

    def definir_resultado_modelo1(self, resultado):
        """Define o resultado do Modelo 1."""
        self.modelo1 = resultado

    def definir_resultado_modelo2(self, resultado):
        """Define o resultado do Modelo 2."""
        self.modelo2 = resultado

    def definir_resultado_modelo3(self, resultado):
        """Define o resultado do Modelo 3."""
        self.modelo3 = resultado

    def obter_resultado_modelo1(self):
        """Retorna o resultado do Modelo 1."""
        return self.modelo1

    def obter_resultado_modelo2(self):
        """Retorna o resultado do Modelo 2."""
        return self.modelo2

    def obter_resultado_modelo3(self):
        """Retorna o resultado do Modelo 3."""
        return self.modelo3

    def obter_todos_resultados(self):
        """Retorna um dicionário com os resultados de todos os modelos."""
        return {
            'Modelo 1': self.modelo1,
            'Modelo 2': self.modelo2,
            'Modelo 3': self.modelo3
        }
        
        

class NFT_Blue_Chips:
    def __init__(self):
        self.lista_df_blue_chips = []  # Atributo para armazenar uma lista de DataFrames

    def adicionar_nft(self, dataframe):
        # Adiciona um novo DataFrame à lista
        if isinstance(dataframe, pd.DataFrame):
            self.lista_df_blue_chips.append(dataframe)
        else:
            novo_df = pd.DataFrame(dataframe)
            self.lista_df_blue_chips.append(novo_df)

    def adicionar_lista_nfts(self, lista_dataframes):
        # Adiciona uma lista de DataFrames à lista existente
        for df in lista_dataframes:
            if isinstance(df, pd.DataFrame):
                self.lista_df_blue_chips.append(df)
            else:
                print("O item na lista não é um DataFrame válido.")

    def exibir_dataframes(self):
        # Exibe todos os DataFrames armazenados na lista
        if self.lista_df_blue_chips:
            for i, df in enumerate(self.lista_df_blue_chips, start=1):
                print(f"DataFrame {i}:")
                print(df, "\n")
        else:
            print("A lista de DataFrames está vazia.")

    def obter_dataframe(self, indice):
        # Retorna um DataFrame específico da lista pelo índice
        if 0 <= indice < len(self.lista_df_blue_chips):
            return self.lista_df_blue_chips[indice]
        else:
            print("Índice fora do alcance.")
            return None

    def obter_lista_completa(self):
        return self.lista_df_blue_chips


class NFT_Comum:
    def __init__(self):
        self.df_nft = None  # Atributo para armazenar o DataFrame
        self.lista_df_comum = []  # Atributo para armazenar uma lista de DataFrames
 
    def carregar_nft(self, dataframe):
        self.df_nft = pd.DataFrame(dataframe)

    def exibir_dataframe(self):
        if self.df_nft is not None:
            # Simulação de exibição do DataFrame
            print(self.df_nft)
        else:
            print("DataFrame está vazio.")
    
    def adicionar_lista_nfts(self, lista_dataframes):
        # Adiciona uma lista de DataFrames à lista existente
        for df in lista_dataframes:
            if isinstance(df, pd.DataFrame):
                self.lista_df_comum.append(df)
            else:
                print("O item na lista não é um DataFrame válido.")
    
    def obter_dataframe(self, indice):
        # Retorna um DataFrame específico da lista pelo índice
        if 0 <= indice < len(self.lista_df_comum):
            return self.lista_df_comum[indice]
        else:
            print("Índice fora do alcance.")
            return None

    def limpar_lista(self):
        self.lista_df_comum = []

    def obter_lista_completa(self):
        return self.lista_df_comum             


class TS_nft_blue:
    def __init__(self):
        """Inicializa a lista vazia."""
        self.ts_nft_blue = []

    def atribuir_lista(self, lista):
        self.ts_nft_blue = lista

    def adicionar_item(self, item):
        """Adiciona um único item à lista."""
        self.ts_nft_blue.append(item)

    def obter_lista(self):
        """Retorna a lista completa."""
        return self.ts_nft_blue

    def obter_item(self, indice):
        """Retorna um item específico da lista pelo índice, se existir."""
        if 0 <= indice < len(self.ts_nft_blue):
            return self.ts_nft_blue[indice]
        else:
            print("Índice fora do alcance.")
            return None
    
class TS_nft_comum:
    def __init__(self):
        """Inicializa a lista vazia."""
        self.ts_nft_comum = []

    def atribuir_lista(self, lista):
        self.ts_nft_comum = lista

    def adicionar_item(self, item):
        """Adiciona um único item à lista."""
        self.ts_nft_comum.append(item)

    def obter_lista(self):
        """Retorna a lista completa."""
        return self.ts_nft_comum

    def obter_item(self, indice):
        """Retorna um item específico da lista pelo índice, se existir."""
        if 0 <= indice < len(self.ts_nft_comum):
            return self.ts_nft_comum[indice]
        else:
            print("Índice fora do alcance.")
            return None

# Funções
   
def adicionar_mensagem(mensagem):
    mensagem_area.config(state=tk.NORMAL)
    mensagem_area.insert(tk.END, mensagem + "\n")
    mensagem_area.config(state=tk.DISABLED)
    mensagem_area.see(tk.END)


def agrupar_dataframes_por_data(df_list_blue, df_list_comum): 
    
    list_grouped_df_blue = []
    list_grouped_df_comum = []
    
    # Loop para agrupar os DataFrames e criar as listas list_grouped_df_blue e list_grouped_df_comum
    for dataframe_list, grouped_list in zip([df_list_blue, df_list_comum], [list_grouped_df_blue, list_grouped_df_comum]):
        for df in dataframe_list:
            grouped_df = df.groupby('BLOCK_DATE')
            grouped_list.append(grouped_df)
    
    return list_grouped_df_blue, list_grouped_df_comum


def avaliar_modelo_dbscan(ts_nft_comum):
  
    dbscan = DBSCAN(eps=0.5, min_samples=5)

    # Listas para armazenar métricas e resultados
    accuracy_scores, precision_scores, recall_scores, f1_scores, true_positive_anomalies = [], [], [], [], []
    accuracy_scores_heatmap, precision_scores_heatmap, recall_scores_heatmap, f1_scores_heatmap = [], [], [], []
    
    for i in range(len(ts_nft_comum)):
        test_data = ts_nft_comum[i]

        # Realize a detecção de anomalias no conjunto de teste usando o modelo DBSCAN treinado anteriormente
        test_clusters = dbscan.fit_predict(test_data)

        # Defina um valor para identificar anomalias
        anomaly_label = -1  # Por convenção, -1 é frequentemente usado para anomalias em DBSCAN

        # Identificar anomalias com base no rótulo de anomalia
        test_anomalies = [1 if label == anomaly_label else 0 for label in test_clusters]

        # Suponha que você tenha um limiar de anomalia
        threshold = 0.1  # Ajuste conforme necessário

        # Calcular as métricas para o conjunto de teste
        test_labels = test_data.iloc[:, 0].values
        test_labels_binary = [1 if label > threshold else 0 for label in test_labels]

        accuracy_test = accuracy_score(test_labels_binary, test_anomalies)
        recall_test = recall_score(test_labels_binary, test_anomalies)
        precision_test = precision_score(test_labels_binary, test_anomalies)
        f1_score_test = f1_score(test_labels_binary, test_anomalies)

        # Armazenar os valores das métricas nas listas
        accuracy_scores.append(accuracy_test)
        precision_scores.append(precision_test)
        recall_scores.append(recall_test)
        f1_scores.append(f1_score_test)

        # Calcular a matriz de confusão para o conjunto de teste
        confusion_test = confusion_matrix(test_labels_binary, test_anomalies)
    
        # Verificar se exite algum registro classificado como analia na matriz de confusão - TP > 0
        tp_found = encontrar_tp(confusion_test)
        true_positive_anomalies.append(tp_found)
    
        # Imprimir a matriz de confusão para os conjuntos de teste
        adicionar_mensagem("\nMatriz de Confusão:")
        ct_str = np.array2string(confusion_test)  # Converte a matriz de confusão para uma string
        adicionar_mensagem(ct_str)

        # Imprimir as métricas para o conjunto de teste
        adicionar_mensagem("\nMétricas:")
        adicionar_mensagem("Acurácia: " + format(accuracy_test, '.4f'))
        adicionar_mensagem("Precisão: " + format(precision_test, '.4f'))
        adicionar_mensagem("Recall: " + format(recall_test, '.4f'))
        adicionar_mensagem("F1-score: " + format (f1_score_test, '.4f'))

    # Calcular as médias das métricas após o loop
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)

    limiar_recall = 0.75  # Definição do limiar de recall desejado como anomalia (75%)

    for tp in true_positive_anomalies:
        if tp == True: 
            # Compara o recall calculado com o limiar desejado
            if average_recall >= limiar_recall:
                adicionar_mensagem("\nO conjunto de dados possui registro(s) classificados como anomalia(s) e possui recall superior ao limiar definido de " + format(limiar_recall *100, '.2f') + "%. Recall: " + format(average_recall *100, '.4f') +"%.")
                modelo3 = 1
            else:
                adicionar_mensagem("\nApesar de possuir registro(s) classificado(s) como anomalia(s) o conjunto de dados não será classificado como anômalo, pois possui um recall abaixo do limiar definido de " + format(limiar_recall *100, '.2f') + "%. Recall: " + format(average_recall *100, '.4f') +"%.")
                modelo3 = 0
        else:
            adicionar_mensagem("\nO conjunto de dados não é classificado como anômalo, pois não há nenhum registro classificado pelo modelo. Recall: " + format(average_recall *100, '.4f') +"%.")
            modelo3 = 0

    return modelo3


def avaliar_modelo_isolation_forest(train_data_blue, ts_nft_comum):
    isolation_forest_model = IsolationForest(contamination='auto', random_state=42)
    
    # Listas para armazenar métricas e resultados
    accuracy_scores, precision_scores, recall_scores, f1_scores, true_positive_anomalies = [], [], [], [], []
    # accuracy_scores_heatmap, precision_scores_heatmap, recall_scores_heatmap, f1_scores_heatmap = [], [], [], []
    
    # Treinar o modelo com os dados de treinamento
    for train_df in train_data_blue:
        isolation_forest_model.fit(train_df)

    # Testar o modelo com ts_nft_comum
    for i, test_data in enumerate(ts_nft_comum):
        test_pred = isolation_forest_model.predict(test_data)
        test_pred_binary = [1 if pred == -1 else 0 for pred in test_pred]
        
        # Supondo que a primeira coluna contém as labels
        test_labels = test_data.iloc[:, 0].values
        test_labels_binary = [1 if label != 0 else 0 for label in test_labels]

        # Calcular métricas
        accuracy_test = accuracy_score(test_labels_binary, test_pred_binary)
        recall_test = recall_score(test_labels_binary, test_pred_binary)
        precision_test = precision_score(test_labels_binary, test_pred_binary)
        f1_score_test = f1_score(test_labels_binary, test_pred_binary)
        confusion_test = confusion_matrix(test_labels_binary, test_pred_binary)

        # Verificar presença de TP
        tp_found = encontrar_tp(confusion_test)
        true_positive_anomalies.append(tp_found)

        # Imprimir resultados
        adicionar_mensagem(f"\nMatriz de Confusão:")
        ct_str = np.array2string(confusion_test)  # Converte a matriz de confusão para uma string
        adicionar_mensagem(ct_str)
        adicionar_mensagem(f"\nMétricas:")
        print_metrics_2(accuracy_test, precision_test, recall_test, f1_score_test)

        # Armazenar métricas
        accuracy_scores.append(accuracy_test)
        precision_scores.append(precision_test)
        recall_scores.append(recall_test)
        f1_scores.append(f1_score_test)

    # Calcular médias das métricas
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)

    limiar_recall = 0.75  # Definição do limiar de recall desejado como anomalia (>=75%)

    for tp in true_positive_anomalies:
        if tp == True: 
            # Compara o recall calculado com o limiar desejado
            if average_recall >= limiar_recall:
                adicionar_mensagem("\nO conjunto de dados possui registro(s) classificados como anomalia(s) e possui recall superior ao limiar definido de " + format(limiar_recall *100, '.2f') + "%. Recall: " + format(average_recall *100, '.4f') +"%.")
                modelo2 = 1
            else:
                adicionar_mensagem("\nApesar de possuir registro(s) classificado(s) como anomalia(s) o conjunto de dados não será classificado como anômalo, pois possui um recall abaixo do limiar definido de " + format(limiar_recall *100, '.2f') + "%. Recall: " + format(average_recall *100, '.4f') +"%.")
                modelo2 = 0
        else:
            adicionar_mensagem("\nO conjunto de dados não é classificado como anômalo, pois não há nenhum registro classificado pelo modelo. Recall: " + format(average_recall *100, '.4f') +"%.")
            modelo2 = 0

    return modelo2


def avaliar_modelo_lof(train_data_blue, ts_nft_comum):
    lof = LocalOutlierFactor(novelty=True, n_neighbors=20, contamination=0.3)
    
    accuracy_scores, precision_scores, recall_scores, f1_scores, true_positive_anomalies = [], [], [], [], []
    # accuracy_scores_heatmap, precision_scores_heatmap, recall_scores_heatmap, f1_scores_heatmap = [], [], [], []

    # Treinar o modelo com os dados de treinamento
    for train_df in train_data_blue:
        lof.fit(train_df)

    # Testar o modelo com ts_nft_comum
    for i, test_data in enumerate(ts_nft_comum):
        test_pred = lof.predict(test_data)
        test_pred_binary = [1 if pred == -1 else 0 for pred in test_pred]
        
        # Supondo que a primeira coluna contém as labels
        test_labels = test_data.iloc[:, 0].values
        test_labels_binary = [1 if label != 0 else 0 for label in test_labels]

        # Calcular métricas
        accuracy_test = accuracy_score(test_labels_binary, test_pred_binary)
        recall_test = recall_score(test_labels_binary, test_pred_binary)
        precision_test = precision_score(test_labels_binary, test_pred_binary)
        f1_score_test = f1_score(test_labels_binary, test_pred_binary)
        confusion_test = confusion_matrix(test_labels_binary, test_pred_binary)

        # Verificar presença de TP
        tp_found = encontrar_tp(confusion_test)
        true_positive_anomalies.append(tp_found)

        # Imprimir resultados
        adicionar_mensagem(f"\nMatriz de Confusão:")
        ct_str = np.array2string(confusion_test)  # Converte a matriz de confusão para uma string
        adicionar_mensagem(ct_str)
        adicionar_mensagem(f"\nMétricas:")
        print_metrics_2(accuracy_test,precision_test,recall_test,f1_score_test)
        
        # Armazenar métricas
        accuracy_scores.append(accuracy_test)
        precision_scores.append(precision_test)
        recall_scores.append(recall_test)
        f1_scores.append(f1_score_test)

    # Calcular médias das métricas
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)

    # Teste de True Positive e voto do modelo LOF
    limiar_recall = 0.75  # Definição do limiar de recall desejado como anomalia (>=75%)

    for tp in true_positive_anomalies:
        if tp == True: 
            # Compara o recall calculado com o limiar desejado
            if average_recall >= limiar_recall:
                adicionar_mensagem("\nO conjunto de dados possui registro(s) classificados como anomalia(s) e possui recall superior ao limiar definido de " + format(limiar_recall *100, '.2f') + "%. Recall: " + format(average_recall *100, '.4f') +"%.")
                modelo1 = 1
            else:
                adicionar_mensagem("\nApesar de possuir registro(s) classificado(s) como anomalia(s) o conjunto de dados não será classificado como anômalo, pois possui um recall abaixo do limiar definido de " + format(limiar_recall *100, '.2f') + "%. Recall: " + format(average_recall *100, '.4f') +"%.")
                modelo1 = 0
        else:
            adicionar_mensagem("\nO conjunto de dados não é classificado como anômalo, pois não há nenhum registro classificado pelo modelo. Recall: " + format(average_recall *100, '.4f') +"%.")
            modelo1 = 0

    return modelo1


def calculate_metrics(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return accuracy, precision, recall, f1


def carregar_credenciais():
    mensagem_area.config(state='normal')
    
    # Autenticação BigQuery
    BQ_client.autenticar_bigquery()
    # client_bigquery = autenticar_bigquery()

    # Ler Chave API Etherscan
    api_key_etherscan = ler_chave_etherscan()
    
    # Carregar a chave no objeto "Api_Key" criado ao carregar o programa
    Api_Key.carregar_chave(api_key_etherscan)

    mensagem_area.config(state='disabled')
    notebook.tab(1, state="normal")  # Habilita a aba "Carga das NFTs Blue Chips"
    notebook.tab(0, state="disabled")  # Desabilita a aba atual
    notebook.select(1)  # Muda para a próxima aba automaticamente
    messagebox.showinfo("Credenciais", "Credenciais carregadas com sucesso. Você pode prosseguir para a próxima etapa.")

    
def carregar_nfts_blue_chips():
    mensagem_area.config(state='normal')
    
    # Carregar os arquivos CSV das NFTs Blue Chips
    adicionar_mensagem("==================== Etapa 2 ====================\n")

    list_csv_blue = []
    list_csv_blue = lista_df()

    nft_blue_chips.adicionar_lista_nfts(list_csv_blue)

    mensagem_area.config(state='disabled')
    notebook.tab(2, state="normal")  # Habilita a aba "Consulta de NFT Comum"
    notebook.tab(1, state="disabled")  # Desabilita a aba atual
    notebook.select(2)  # Muda para a próxima aba automaticamente
    messagebox.showinfo("NFTs Blue Chips", "NFTs Blue Chips carregadas com sucesso. Você pode prosseguir para a consulta de NFT comum.")

    
def consultar_nft_comum():

    adicionar_mensagem("==================== Etapa 3 ====================\n")
    mensagem_area.config(state='normal')
    
    list_nft_comum = []
    api_key_etherscan = Api_Key.obter_chave()
    hash_nft = entrada_nft_comum.get()
    num_transactions = verificar_transacoes_nft(hash_nft, api_key_etherscan)
    
    # Garante que o valor retornado seja convertido para int
    qtde_transactions = int(num_transactions)

    if qtde_transactions > 1500:
        adicionar_mensagem(f"Realizando coleta de transações da NFT {hash_nft} na GCP.\n")

        # Chamar a função para executar a consulta BigQuery e obter o DataFrame resultante
        resultado_df = executar_consulta_bigquery(hash_nft)
        
        list_nft_comum.append(resultado_df)
        print(resultado_df.info())
        
        nft_comum.carregar_nft(resultado_df)
        nft_comum.adicionar_lista_nfts(list_nft_comum)
   
        # Muda a configuração da aba e exibe a mensagem somente se a condição for satisfeita
        notebook.tab(3, state="normal")  # Habilita a aba "Pre-Processamento"
        notebook.tab(2, state="disabled")  # Desabilita a aba atual
        notebook.select(3)  # Muda para a próxima aba automaticamente
        messagebox.showinfo("NFT Comum", "Etapa de coleta de informações das transações da NFT concluída.")
    else:
        # Adicione o comportamento desejado se a quantidade de transações for <= 1500
        messagebox.showinfo("NFT Comum", "Quantidade de transações insuficiente para prosseguir.")


def criar_sumario_dataframes(list_grouped_df_comum, list_grouped_df_blue):
        
    ts_nft_comum = []  # Lista para armazenar os DataFrames criados a partir de list_grouped_df_comum
    ts_nft_blue = []   # Lista para armazenar os DataFrames criados a partir de list_grouped_df_blue

    # Loop para iterar sobre list_grouped_df_comum e list_grouped_df_blue
    for label, grouped_list, ts_nft_list in [("Comum", list_grouped_df_comum, ts_nft_comum), ("Blue Chips", list_grouped_df_blue, ts_nft_blue)]:
        adicionar_mensagem(f"Elaboração das Série(s) Temporal(is) Multivariada(s) - NFT {label}")
    
        for i, ts in enumerate(grouped_list):
   
            # Criar um novo DataFrame vazio para cada grupo
            df = pd.DataFrame()
        
            # Adicionar a coluna 'BLOCK_DATE' ao DataFrame
            df['BLOCK_DATE'] = ts['BLOCK_DATE'].mean()
        
            # Número de transações diárias
            df['QTDE_TRANSACOES_DIA'] = ts['BLOCK_DATE'].count()
        
            # Média de GAS diário
            df['MEDIA_GAS_DIA'] = ts['GAS'].mean()
        
            # Média de GAS_LIMIT diário
            df['MEDIA_GAS_LIMIT_DIA'] = ts['RECEIPT_GAS_USED'].mean()
        
            # Quantidade de compradores únicos diários
            df['QTDE_COMPRADORES_UNICOS_DIA'] = ts['FROM_ADDRESS'].nunique()
        
            # Quantidade de vendedores únicos diários
            df['QTDE_VENDEDORES_UNICOS_DIA'] = ts['TO_ADDRESS'].nunique()
        
            # Cálculo de NOVOS_TITULARES
            df['NOVOS_TITULARES'] = (ts['FROM_ADDRESS'].nunique() + ts['TO_ADDRESS'].nunique()) / (ts['FROM_ADDRESS'].count() + ts['TO_ADDRESS'].count()) 
        
            # Transforma a coluna 'BLOCK_DATE' em índice temporal
            df.set_index('BLOCK_DATE', inplace=True)
        
            # Adicionar o DataFrame criado à lista correspondente
            ts_nft_list.append(df)

    return ts_nft_comum, ts_nft_blue


def decidir_anomalia(modelo1, modelo2, modelo3):
    # Contagem de votos
    contagem_anomalia = modelo1 + modelo2 + modelo3
    contagem_nao_anomalia = 3 - contagem_anomalia

    # Decisão de votação (usando uma maioria simples)
    limite_votacao = 2  # Pelo menos 2 modelos devem votar como anomalia
    if contagem_anomalia >= limite_votacao:
        decisao_final = "Conjunto de dados Anômalo/Suspeito"
    else:
        decisao_final = "Conjunto de dados Normal/Regular"

    return "Decisão Final: " + decisao_final


def dividir_dataframes(ts_nft_blue):
    train_data_blue, validation_data_blue, test_data_blue  = [], [], []

    for df in ts_nft_blue:
        # Dividir o DataFrame em treinamento (60%), validação (20%) e teste (20%)
        train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
        validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Adicionar os DataFrames resultantes às listas correspondentes
        train_data_blue.append(train_df)
        validation_data_blue.append(validation_df)
        test_data_blue.append(test_df)
    
    return train_data_blue, validation_data_blue, test_data_blue


def encontrar_tp(confusion_matrix_val):
    tp_found = confusion_matrix_val[0, 0] > 0
    return tp_found


def executar_consulta_bigquery(nft_hash):
    client = BQ_client.get_client()
           
    adicionar_mensagem(f"Obtenção Transações NFT - Plataforma Google BigQuery.")

    # SQL - Consulta com o valor dinâmico de nft_hash
    query = f"""
    SELECT
      A.BLOCK_TIMESTAMP,
      A.FROM_ADDRESS,
      A.TO_ADDRESS,
      A.VALUE,
      A.TRANSACTION_HASH,
      B.NONCE,
      B.FROM_ADDRESS AS FROM_ADDRESS_BLOCKCHAIN,
      B.TO_ADDRESS AS TO_ADDRESS_BLOCKCHAIN,
      B.GAS,
      B.RECEIPT_GAS_USED
    FROM
      `bigquery-public-data.crypto_ethereum.token_transfers` AS A
    INNER JOIN
      `bigquery-public-data.crypto_ethereum.transactions` AS B
    ON
      A.transaction_hash = B.HASH
    WHERE
      A.TOKEN_ADDRESS = '{nft_hash}'
      AND A.BLOCK_TIMESTAMP >= (
      SELECT
        MIN(block_timestamp)
      FROM
        `bigquery-public-data.crypto_ethereum.contracts`
      WHERE
        address = '{nft_hash}')
    ORDER BY
      A.BLOCK_TIMESTAMP
    """
    
    # Execução da consulta SQL
    query_job = client.query(query)

    # Extrair os resultados como uma lista de dicionários
    results = []
    for row in query_job:
        results.append(dict(row.items()))

    # Criar um DataFrame a partir dos resultados
    result_df = pd.DataFrame(results)

    return result_df        
 
def executar_modelos_am():

    adicionar_mensagem("==================== Etapa 5 ====================\n")
    mensagem_area.config(state='normal')
 
    # Ações para execução dos modelos de AM
    # Objetos ts_nft_blue e ts_nft_comum com as listas necessárias
    
    # Dividir cada DataFrame em ts_nft_blue em conjuntos de treinamento, validação e teste
    train_data_blue, validation_data_blue, test_data_blue = [], [], []
    
    adicionar_mensagem(f"Divisão das NFTs Blue Chips - Treinamento, Validação e Teste.")

    train_data_blue, validation_data_blue, test_data_blue = dividir_dataframes(ts_nft_blue.obter_lista())


    # LOF
    adicionar_mensagem(f"\nUtilização modelo de AM - LOF.")
    adicionar_mensagem(f"Treino, Validação e Testes do modelo LOF - NFTs Blue Chips.")

    treinar_validar_testar_lof(train_data_blue, validation_data_blue, test_data_blue)

    modelo1 = avaliar_modelo_lof(train_data_blue, ts_nft_comum.obter_lista())
    
    comite_AM.definir_resultado_modelo1(int(modelo1))
        
   # Isolation Forest
    adicionar_mensagem(f"\nUtilização modelo de AM - Isolation Forest.")
    adicionar_mensagem(f"Treino, Validação e Testes do modelo Isolation Forest - NFTs Blue Chips.")

    treinar_validar_testar_isolation_forest(train_data_blue, validation_data_blue, test_data_blue)

    modelo2 = avaliar_modelo_isolation_forest(train_data_blue, ts_nft_comum.obter_lista())

    comite_AM.definir_resultado_modelo2(int(modelo2))

    # DBScan
    adicionar_mensagem(f"\nUtilização modelo de AM - DBScan.")
    adicionar_mensagem(f"Treino, Validação e Testes do modelo DBScan - NFTs Blue Chips.")

    treinar_validar_testar_dbscan(train_data_blue, validation_data_blue, test_data_blue)

    modelo3 = avaliar_modelo_dbscan(ts_nft_comum.obter_lista())

    comite_AM.definir_resultado_modelo3(int(modelo3))
    
    mensagem_area.config(state='disabled')
    notebook.tab(5, state="normal")  # Habilita a aba "Executar Modelos de AM"
    notebook.tab(4, state="disabled")  # Desabilita a aba atual
    notebook.select(5)  # Muda para a próxima aba automaticamente
    messagebox.showinfo("Modelos de AM", "Execução dos modelos de AM concluída. Você pode prosseguir para a etapa de votação do comitê de classificação.")

    
def ler_chave_etherscan():
    config_file = selecionar_arquivo_configuracao()

    if config_file and os.path.isfile(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        api_key = config.get('API_KEYS', 'ETHERSCAN_API_KEY', fallback=None)
        if not api_key:
            adicionar_mensagem("Chave da API do Etherscan não encontrada no arquivo de configuração.")
        else:
            adicionar_mensagem("Chave da API do Etherscan carregada com sucesso.\n")
        return api_key
    else:
        adicionar_mensagem("O arquivo de configuração não foi selecionado ou não é válido.")
        return None   


def limitar_tamanho(*args):
    valor = texto_entrada.get()
    if len(valor) > 42:  # Limite de 42 caracteres
        texto_entrada.set(valor[:42])
    # Habilita o botão "Consultar" somente se o campo de entrada estiver preenchido
    botao_consultar_nft_comum['state'] = tk.NORMAL if valor else tk.DISABLED
        
def limpar_entrada():
    texto_entrada.set("")
    
    
def lista_df():
    try:
        # Permite ao usuário selecionar o diretório onde os arquivos CSV estão localizados
        config_dir = selecionar_diretorio_csv()
        if not config_dir:
            adicionar_mensagem("Nenhum diretório selecionado.")
            return []

        adicionar_mensagem(f"Diretório selecionado: {config_dir}")

        # Lista vazia para armazenar os dataframes
        df_list = []

        # Contador para os arquivos carregados
        files_loaded = 0

        # Localizar todos os arquivos CSV no diretório selecionado
        for filename in glob.glob(os.path.join(config_dir, "*.csv")):
            try:
                # Ler o arquivo CSV em um dataframe
                df = pd.read_csv(filename)
                # Adicionar o dataframe à lista
                df_list.append(df)
                files_loaded += 1
            except Exception as e:
                adicionar_mensagem(f"Erro ao carregar o arquivo: {filename}, Erro: {str(e)}")

        adicionar_mensagem(f"Arquivos CSV carregados: {files_loaded}\n")
        return df_list
    except Exception as e:
        adicionar_mensagem(f"Erro na função lista_df: {str(e)}")
        return []


def normalize_dataframes(dataframe_list):
    # Inicialize o MinMaxScaler
    scaler = MinMaxScaler()

    for idx, ts in enumerate(dataframe_list):
        # Selecione apenas as colunas numéricas para normalização
        columns_to_normalize = [
            'QTDE_TRANSACOES_DIA',
            'MEDIA_GAS_DIA',
            'MEDIA_GAS_LIMIT_DIA',
            'QTDE_COMPRADORES_UNICOS_DIA',
            'QTDE_VENDEDORES_UNICOS_DIA',
            'NOVOS_TITULARES'
        ]
        # Aplique o Min-Max Scaling às colunas selecionadas
        dataframe_list[idx][columns_to_normalize] = scaler.fit_transform(ts[columns_to_normalize])


def pre_processamento():
    
    adicionar_mensagem("==================== Etapa 4 ====================\n")

    mensagem_area.config(state='normal')

    # Ações para execução de pre-processamento
    # Processando a lista de DataFrames list_df_blue
    adicionar_mensagem(f"Pré-Processamento das Informações de NFTs Blue Chips & Comum.\n")

    df_list_blue = []
    for df_blue in nft_blue_chips.obter_lista_completa():
        process_dataframe(df_blue)
        df_list_blue.append(df_blue)

    # Processando a lista de DataFrames list_df_comum
    df_list_comum = []
    for df_comum in nft_comum.obter_lista_completa():
        process_dataframe(df_comum)
        df_list_comum.append(df_comum)
 
    # Agrupamento dos dados utilizando a coluna BLOCK_DATE
    adicionar_mensagem(f"Agrupamento das Informações das NFTs Blue Chips & Comum.\n")

    list_grouped_df_blue, list_grouped_df_comum = [], []

    # Chamada da função agrupar_dataframes_por_data
    list_grouped_df_blue, list_grouped_df_comum = agrupar_dataframes_por_data(df_list_blue, df_list_comum)
  
    # Criação das Series Temporais Multivariadas
    list_ts_nft_comum, list_ts_nft_blue = [], []  

    # Chamada da função criar_sumario_dataframes
    list_ts_nft_comum, list_ts_nft_blue = criar_sumario_dataframes(list_grouped_df_comum, list_grouped_df_blue)

    adicionar_mensagem(f"Normalização das Séries Temporais Multivariadas de NFT Blue Chips & Comum.\n")

    # Normalização das Séries Temporais Multivariadas ts_nft_blue
    normalize_dataframes(list_ts_nft_blue)

    # Normalização da Série Temporal Multivariada ts_nft_comum
    normalize_dataframes(list_ts_nft_comum)

    # Carregar as classes de TS de Blue Chips e Comum
    ts_nft_blue.atribuir_lista(list_ts_nft_blue)
    ts_nft_comum.atribuir_lista(list_ts_nft_comum)
    
    mensagem_area.config(state='disabled')
    notebook.tab(4, state="normal")  # Habilita a aba "Executar Modelos de AM"
    notebook.tab(3, state="disabled")  # Desabilita a aba atual
    notebook.select(4)  # Muda para a próxima aba automaticamente
    messagebox.showinfo("Pre-processamento", "Pré-processamento dos dados concluído. Você pode prosseguir para a etapa de execução dos modelos de AM.")


def print_metrics(metrics_mean):
    # Acurácia
    acuracia_mensagem = "Acurácia: {:.4f}".format(metrics_mean[0])
    adicionar_mensagem(acuracia_mensagem)

    # Precisão
    precisao_mensagem = "Precisão: {:.4f}".format(metrics_mean[1])
    adicionar_mensagem(precisao_mensagem)

    # Recall
    recall_mensagem = "Recall: {:.4f}".format(metrics_mean[2])
    adicionar_mensagem(recall_mensagem)

    # F1-score
    f1_score_mensagem = "F1-score: {:.4f}".format(metrics_mean[3])
    adicionar_mensagem(f1_score_mensagem)


def print_metrics_2(accuracy, precision, recall, f1_score):
    # Acurácia
    acuracia_mensagem = "Acurácia: {:.4f}".format(accuracy)
    adicionar_mensagem(acuracia_mensagem)

    # Precisão
    precisao_mensagem = "Precisão: {:.4f}".format(precision)
    adicionar_mensagem(precisao_mensagem)

    # Recall
    recall_mensagem = "Recall: {:.4f}".format(recall)
    adicionar_mensagem(recall_mensagem)

    # F1-score
    f1_score_mensagem = "F1-score: {:.4f}".format(f1_score)
    adicionar_mensagem(f1_score_mensagem)
    

def process_dataframe(df):
    # Convertendo a coluna 'BLOCK_TIMESTAMP' e 'BLOCK_TIMESTAMP_BLUE' para o tipo datetime
    df['BLOCK_TIMESTAMP'] = pd.to_datetime(df['BLOCK_TIMESTAMP'])

    # Criando uma nova coluna com a data (sem o horário) dos blocos
    df['BLOCK_DATE'] = df['BLOCK_TIMESTAMP'].dt.date

    # Convertendo a coluna 'BLOCK_DATE' para o tipo datetime
    df['BLOCK_DATE'] = pd.to_datetime(df['BLOCK_DATE'])
    
    
def selecionar_arquivo_configuracao():
    # Abre o diálogo para escolher o arquivo
    filepath = filedialog.askopenfilename(
        title="Selecione o arquivo de configuração da API do Etherscan",
        filetypes=(("Arquivos INI", "*.ini"), ("Todos os arquivos", "*.*"))
    )
    return filepath


def selecionar_arquivo_credenciais():
    # Abre o diálogo para escolher o arquivo
    filepath = filedialog.askopenfilename(
        title="Selecione o arquivo de credenciais",
        filetypes=(("Arquivos JSON", "*.json"), ("Todos os arquivos", "*.*"))
    )
    return filepath    
    
def selecionar_diretorio_csv():
    root = tk.Tk()
    root.withdraw()  # Oculta a janela root
    diretorio = filedialog.askdirectory(title="Selecione o diretório contendo os arquivos CSV")
    root.destroy()
    return diretorio

def treinar_validar_testar_dbscan(train_data_blue, validation_data_blue, test_data_blue):
    # Configuração do DBSCAN
    eps = 0.5  # O raio máximo entre dois pontos para serem considerados no mesmo bairro
    min_samples = 5  # O número mínimo de pontos para formar um cluster denso
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Listas para armazenar métricas e resultados
    accuracy_scores, precision_scores, recall_scores, f1_scores, true_positive_anomalies = [], [], [], [], []
    
    # Treinar o modelo com os dados de treinamento
    for train_df in train_data_blue:
        dbscan.fit(train_df)
        
    # Validar o modelo com os dados de validação
    for validation_df in validation_data_blue:
        validation_labels = dbscan.fit_predict(validation_df)

        # Suponha que os pontos atribuídos ao rótulo -1 são anomalias
        validation_predictions = [1 if label == -1 else 0 for label in validation_labels]

        # Suponha que todos os exemplos de validação são normais
        validation_true_labels = np.ones(len(validation_df))

        # Calcular a matriz de confusão
        confusion = confusion_matrix(validation_true_labels, validation_predictions)

        # Calcular os indicadores
        accuracy_valid = accuracy_score(validation_true_labels, validation_predictions)
        precision_valid = precision_score(validation_true_labels, validation_predictions)
        recall_valid = recall_score(validation_true_labels, validation_predictions)
        f1_valid = f1_score(validation_true_labels, validation_predictions)

        # Armazenar os resultados dos indicadores
        accuracy_scores.append(accuracy_valid)
        precision_scores.append(precision_valid)
        recall_scores.append(recall_valid)
        f1_scores.append(f1_valid)
      
    # Calcular a média dos indicadores
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)

    for lista in (accuracy_scores, precision_scores, recall_scores, f1_scores):
        while lista:
            lista.pop()
        
    # Validar o modelo com os dados de teste
    for test_df in test_data_blue:
        test_labels = dbscan.fit_predict(test_df)

        # Suponha que os pontos atribuídos ao rótulo -1 são anomalias
        test_predictions = [1 if label == -1 else 0 for label in test_labels]

        # Suponha que todos os exemplos de teste são normais
        test_true_labels = np.ones(len(test_df))

        # Calcular a matriz de confusão
        confusion = confusion_matrix(test_true_labels, test_predictions)

        # Calcular os indicadores
        accuracy_test = accuracy_score(test_true_labels, test_predictions)
        precision_test = precision_score(test_true_labels, test_predictions)
        recall_test = recall_score(test_true_labels, test_predictions)
        f1_test = f1_score(test_true_labels, test_predictions)

        # Armazenar os resultados dos indicadores
        accuracy_scores.append(accuracy_test)
        precision_scores.append(precision_test)
        recall_scores.append(recall_test)
        f1_scores.append(f1_test)

              
    # Calcular a média dos indicadores
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)


def treinar_validar_testar_isolation_forest(train_data_blue, validation_data_blue, test_data_blue):
    # Inicializar o modelo Isolation Forest
    isolation_forest_model = IsolationForest(contamination='auto', random_state=42)

    # Treinar o modelo com os dados de treinamento
    for train_df in train_data_blue:
        isolation_forest_model.fit(train_df)

    # Validar o modelo com os dados de validação
    validation_metrics_list = []
    validation_confusion_matrix_list = []

    # Processar validação
    for validation_df in validation_data_blue:
        validation_predictions = isolation_forest_model.predict(validation_df)
        validation_true_labels = np.ones(len(validation_df))
        validation_metrics = calculate_metrics(validation_predictions, validation_true_labels)
        validation_metrics_list.append(validation_metrics)
        validation_confusion_matrix_list.append(confusion_matrix(validation_true_labels, validation_predictions))

    # Calcular e imprimir média dos indicadores para validação
    validation_metrics_mean = np.mean(validation_metrics_list, axis=0)

    # Processar teste
    test_metrics_list = []
    test_confusion_matrix_list = []

    for test_df in test_data_blue:
        test_predictions = isolation_forest_model.predict(test_df)
        test_true_labels = np.ones(len(test_df))
        test_metrics = calculate_metrics(test_predictions, test_true_labels)
        test_metrics_list.append(test_metrics)
        test_confusion_matrix_list.append(confusion_matrix(test_true_labels, test_predictions))

    # Calcular e imprimir média dos indicadores para teste
    test_metrics_mean = np.mean(test_metrics_list, axis=0)


def treinar_validar_testar_lof(train_data_blue, validation_data_blue, test_data_blue):
    # Inicializar o modelo LOF

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

    # Treinar o modelo com os dados de treinamento
    for train_df in train_data_blue:
        lof.fit(train_df)

    # Validar o modelo com os dados de validação
    validation_metrics_list = []
    validation_confusion_matrix_list = []

    # Testar o modelo com os dados de teste
    test_metrics_list = []
    test_confusion_matrix_list = []


    # Processar validação
    for validation_df in validation_data_blue:
        validation_predictions = lof.fit_predict(validation_df)
        validation_true_labels = np.ones(len(validation_df))
        validation_metrics = calculate_metrics(validation_predictions, validation_true_labels)
        validation_metrics_list.append(validation_metrics)
        validation_confusion_matrix_list.append(confusion_matrix(validation_true_labels, validation_predictions))

    # Calcular e imprimir média dos indicadores para validação
    validation_metrics_array = np.array(validation_metrics_list)
    validation_metrics_mean = validation_metrics_array.mean(axis=0)

    # Processar teste
    for test_df in test_data_blue:
        test_predictions = lof.fit_predict(test_df)
        test_true_labels = np.ones(len(test_df))
        test_metrics = calculate_metrics(test_predictions, test_true_labels)
        test_metrics_list.append(test_metrics)
        test_confusion_matrix_list.append(confusion_matrix(test_true_labels, test_predictions))

    # Calcular e imprimir média dos indicadores para teste
    test_metrics_array = np.array(test_metrics_list)
    test_metrics_mean = test_metrics_array.mean(axis=0)

def verificar_transacoes_nft(nft_hash, api_key):
    adicionar_mensagem(f"Consulta Quantidade Transações NFT - Plataforma Etherscan.")
    
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={nft_hash}&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["status"] == "1":
            num_transactions = len(data["result"])
            if num_transactions > 1500:
                msg = f"A NFT possui mais de 1500 transações."
                adicionar_mensagem(msg)
            else:
                msg = f"A NFT não atende ao critério de mais de 1500 transações. \nQuantidade de transações: {num_transactions}"
                adicionar_mensagem(msg)
            return num_transactions
        else:
            adicionar_mensagem("Ocorreu um erro na solicitação à API do Etherscan.")
    except Exception as e:
        adicionar_mensagem(f"Erro ao fazer a solicitação à API do Etherscan: {str(e)}")
    
def votacao_comite():

    adicionar_mensagem("\n==================== Etapa 6 ====================\n")

    mensagem_area.config(state='normal')
    
    #Comite de Classificação
    adicionar_mensagem(f"Comitê de Classificação de NFT.")
    resultado = decidir_anomalia(comite_AM.obter_resultado_modelo1(), comite_AM.obter_resultado_modelo2(), comite_AM.obter_resultado_modelo3())
    adicionar_mensagem(f"{resultado}\n")
    messagebox.showinfo("Votação Comitê", "Classificação da NFT pelo Comitê concluída.")
    
    nft_comum.limpar_lista()
    #
    mensagem_area.config(state='disabled')
    notebook.tab(2, state="normal")  # Habilita a aba "Consultar NFT Comum"
    notebook.tab(5, state="disabled")  # Desabilita a aba atual
    notebook.select(2)  # Muda para a próxima aba de "Consultar NFT Comum"

   
############################# ARQUIVO DE TESTES ##############################################################    

def teste_arquivo_nft_comum():

    adicionar_mensagem("==================== Etapa 3 (Teste Arquivo) ====================\n")

    # Chama a função para obter o caminho do arquivo CSV selecionado
    print("\nARQUIVO DE TESTE NFT COMUM\n")
    nft_file = teste_selecionar_arquivo_nft_comum()
    
    if nft_file:  # Verifica se um caminho de arquivo foi selecionado
        try:
            list_nft_comum = []
            # Carrega o arquivo CSV em um DataFrame
            resultado_df = pd.read_csv(nft_file)
            print("Arquivo CSV de testes carregado com sucesso.")
            # Exibe as primeiras linhas do DataFrame para verificação
            # print(f"Tipo resultado_df: {type(resultado_df)}")
            print(resultado_df.info())
            adicionar_mensagem(f"Realizando coleta de transações da NFT no arquivo fornecido.\n")
            
            list_nft_comum.append(resultado_df)
            print(resultado_df.head())

            nft_comum.carregar_nft(resultado_df)
            nft_comum.adicionar_lista_nfts(list_nft_comum)

            # Muda a configuração da aba e exibe a mensagem somente se a condição for satisfeita
            notebook.tab(3, state="normal")  # Habilita a aba "Pre-Processamento"
            notebook.tab(2, state="disabled")  # Desabilita a aba atual
            notebook.select(3)  # Muda para a próxima aba automaticamente
            messagebox.showinfo("NFT Comum", "Etapa de coleta de informações das transações da NFT concluída.") 
            
        except Exception as e:
            print(f"Erro ao carregar o arquivo de teste: {e}")
    else:
        print("Nenhum arquivo foi selecionado.")

def teste_selecionar_arquivo_nft_comum():
    # Abre o diálogo para escolher o arquivo
    filepath = filedialog.askopenfilename(
        title="Selecione o arquivo de testes de NFT Comum",
        filetypes=(("Arquivos CSV", "*.csv"), ("Todos os arquivos", "*.*"))
    )
    return filepath   
    
# Função principal da GUI
def main():    
    
    global mensagem_area, messagebox, notebook, texto_entrada, entrada_nft_comum, botao_consultar_nft_comum

    # Inicialização da janela principal
    root = tk.Tk()
    root.title("Processo Integrado de Consulta e Classificação de NFTs")
    root.geometry("800x350")

    # Criação do Notebook (Aba)
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill='both', padx=10, pady=10)

    # Criação das abas
    aba_credenciais = ttk.Frame(notebook)
    aba_nfts_blue_chips = ttk.Frame(notebook)
    aba_consulta_nft_comum = ttk.Frame(notebook)
    aba_pre_processamento = ttk.Frame(notebook)
    aba_execucao_modelos = ttk.Frame(notebook)
    aba_comite = ttk.Frame(notebook)

    notebook.add(aba_credenciais, text="Carga das Credenciais")
    notebook.add(aba_nfts_blue_chips, text="Carga das NFTs Blue Chips")
    notebook.add(aba_consulta_nft_comum, text="Consulta de NFT Comum")
    notebook.add(aba_pre_processamento, text="Pré-Processamento") 
    notebook.add(aba_execucao_modelos, text="Execução Modelos AM")
    notebook.add(aba_comite, text="Comitê Classificação")
        
    # Inicialmente, desabilita as abas exceto a primeira
    notebook.tab(1, state="disabled")
    notebook.tab(2, state="disabled")
    notebook.tab(3, state="disabled")
    notebook.tab(4, state="disabled")
    notebook.tab(5, state="disabled")


    # Adicionando conteúdo às abas

    # Aba Carga das Credenciais
    botao_carregar_credenciais = ttk.Button(aba_credenciais, text="Carregar Credenciais", command=carregar_credenciais)
    botao_carregar_credenciais.pack(pady=20)

    # Aba Carga das NFTs Blue Chips
    botao_carregar_nfts_blue_chips = ttk.Button(aba_nfts_blue_chips, text="Carregar NFTs Blue Chips", command=carregar_nfts_blue_chips)
    botao_carregar_nfts_blue_chips.pack(pady=20)

    # Aba Consulta de NFT Comum
    # Variável StringVar com rastreador para limitar o tamanho de entrada
    texto_entrada = tk.StringVar()
    texto_entrada.trace("w", limitar_tamanho)

    instrucoes_label = tk.Label(aba_consulta_nft_comum, text="Digite o código hash da NFT:")
    instrucoes_label.pack(pady=10)
    
    # Campo de entrada com limite de caracteres
    entrada_nft_comum = tk.Entry(aba_consulta_nft_comum, textvariable=texto_entrada, width=50)
    entrada_nft_comum.pack(pady=10)

    # Contêiner para os botões
    frame_botoes = tk.Frame(aba_consulta_nft_comum)
    frame_botoes.pack(pady=10)

    # Botão para consultar NFT Comum (placeholder para a função de consulta)
    botao_consultar_nft_comum = ttk.Button(frame_botoes, text="Consultar", command=consultar_nft_comum, state=tk.DISABLED) 
    # botao_consultar_nft_comum = ttk.Button(frame_botoes, text="Consultar", command=teste_arquivo_nft_comum, state=tk.DISABLED)
    botao_consultar_nft_comum.pack(side=tk.LEFT, padx=5) 
    
    # Botão para limpar os valores do campo de entrada
    botao_limpar = ttk.Button(frame_botoes, text="Limpar", command=limpar_entrada)
    botao_limpar.pack(side=tk.LEFT, padx=5)
    
    # Aba Pre-Processamento
    botao_pre_processamento = ttk.Button(aba_pre_processamento, text="Pré-processamento das Transações", command=pre_processamento)
    botao_pre_processamento.pack(pady=20)
    
    # Aba Execução dos Modelos de AM
    botao_execucao_modelos_am = ttk.Button(aba_execucao_modelos, text="Modelos de AM", command=executar_modelos_am)
    botao_execucao_modelos_am.pack(pady=20)
        
    # Aba Comitê de Classificação
    botao_execucao_comite = ttk.Button(aba_comite, text="Votação Comitê", command=votacao_comite)
    botao_execucao_comite.pack(pady=20)
        
    # Área de visualização de mensagens
    mensagem_area = scrolledtext.ScrolledText(root, height=10)
    mensagem_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    mensagem_area.config(state='disabled')  # Desabilita edição para manter como uma área de log

    root.mainloop()    
        
# Execução do programa
if __name__ == "__main__":
    
    # Objetos das Classes API e GGBQ
    Api_Key = API_Key_Etherscan()
    BQ_client = BigQueryClient()

    nft_comum = NFT_Comum()
    nft_blue_chips = NFT_Blue_Chips()
    
    ts_nft_comum = TS_nft_comum()
    ts_nft_blue = TS_nft_blue()
    
    comite_AM = Comite_AM()
    
    main()