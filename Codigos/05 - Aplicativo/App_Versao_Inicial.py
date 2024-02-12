#!/usr/bin/env python
# coding: utf-8

from tkinter import messagebox, scrolledtext, filedialog
from google.cloud import bigquery
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.cluster import DBSCAN
from tensorflow import keras
from tensorflow.keras import layers, models
import configparser
import requests
import sys
import os
import google
import configparser
import glob
import pandas as pd
import numpy as np
import tkinter as tk



def obter_caminho_absoluto(arquivo_rel):
    if getattr(sys, 'frozen', False):
        # Executando como um executável PyInstaller
        return os.path.abspath(os.path.join(sys._MEIPASS, arquivo_rel))
    else:
        # Executando normalmente como script Python
        return os.path.abspath(os.path.join(os.path.dirname(__file__), arquivo_rel))

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

        adicionar_mensagem(f"Arquivos CSV carregados: {files_loaded}")
        return df_list
    except Exception as e:
        adicionar_mensagem(f"Erro na função lista_df: {str(e)}")
        return []


def autenticar_bigquery():
    credentials_file = selecionar_arquivo_credenciais()
    
    if credentials_file and os.path.isfile(credentials_file):
        try:
            client = bigquery.Client.from_service_account_json(credentials_file)
            adicionar_mensagem("Autenticação bem-sucedida.")
            return client
        except Exception as e:
            adicionar_mensagem(f"Erro de autenticação: {str(e)}")
            return None
    else:
        adicionar_mensagem("O arquivo de credenciais não foi selecionado ou não é válido.")
        return None

def ler_chave_etherscan():
    config_file = selecionar_arquivo_configuracao()

    if config_file and os.path.isfile(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        api_key = config.get('API_KEYS', 'ETHERSCAN_API_KEY', fallback=None)
        if not api_key:
            adicionar_mensagem("Chave da API do Etherscan não encontrada no arquivo de configuração.")
        else:
            adicionar_mensagem("Chave da API do Etherscan carregada com sucesso.")
        return api_key
    else:
        adicionar_mensagem("O arquivo de configuração não foi selecionado ou não é válido.")
        return None


def consultar_nft():
    hash_nft = entrada_hash.get()
    messagebox.showinfo("Consulta NFT", f"Consultando NFT com hash: {hash_nft}")
    verificar_transacoes_nft(hash_nft, api_key_etherscan)

def limpar_entrada():
    entrada_hash.delete(0, tk.END)

def adicionar_mensagem(mensagem):
    text_area.config(state=tk.NORMAL)
    text_area.insert(tk.END, mensagem + "\n")
    text_area.config(state=tk.DISABLED)
    text_area.see(tk.END)

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
                executar_consulta_bigquery(nft_hash)
            else:
                msg = f"A NFT possui {num_transactions} transações, que não atendem ao critério de mais de 1500."
            adicionar_mensagem(msg)
            return num_transactions
        else:
            adicionar_mensagem("Ocorreu um erro na solicitação à API do Etherscan.")
    except Exception as e:
        adicionar_mensagem(f"Erro ao fazer a solicitação à API do Etherscan: {str(e)}")

        
def executar_consulta_bigquery(nft_hash):
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
    adicionar_mensagem(f"Query: {query}")

    # Execução da consulta SQL
    # query_job = client.query(query)

    # Extrair os resultados como uma lista de dicionários
    # results = []
    # for row in query_job:
        # results.append(dict(row.items()))

    # Criar um DataFrame a partir dos resultados
    # import pandas as pd
    # result_df = pd.DataFrame(results)

    # return result_df        
        
def limitar_tamanho(*args):
    valor = texto_entrada.get()
    if len(valor) > 46:
        texto_entrada.set(valor[:46])


def carregar_arquivo_csv():
    root = tk.Tk()
    root.withdraw()  # Oculta a janela root
    caminho_arquivo_csv = filedialog.askopenfilename(
        title="Selecione o arquivo CSV",
        filetypes=[("Arquivos CSV", "*.csv"), ("Todos os arquivos", "*.*")]
    )
    root.destroy()

    if not caminho_arquivo_csv:
        print("Nenhum arquivo selecionado.")
        return None

    try:
        # Lê o arquivo CSV e cria um DataFrame
        df = pd.read_csv(caminho_arquivo_csv)
        adicionar_mensagem(f"Arquivos CSV - NFT Comum - carregado")
        return df
    except Exception as e:
        print(f"Erro ao carregar o arquivo CSV: {str(e)}")
        return None

    
def process_dataframe(df):
    # Convertendo a coluna 'BLOCK_TIMESTAMP' e 'BLOCK_TIMESTAMP_BLUE' para o tipo datetime
    df['BLOCK_TIMESTAMP'] = pd.to_datetime(df['BLOCK_TIMESTAMP'])

    # Criando uma nova coluna com a data (sem o horário) dos blocos
    df['BLOCK_DATE'] = df['BLOCK_TIMESTAMP'].dt.date

    # Convertendo a coluna 'BLOCK_DATE' para o tipo datetime
    df['BLOCK_DATE'] = pd.to_datetime(df['BLOCK_DATE'])


def agrupar_dataframes_por_data(df_list_blue, df_list_comum):
    
    list_grouped_df_blue = []
    list_grouped_df_comum = []
    
    # Loop para agrupar os DataFrames e criar as listas list_grouped_df_blue e list_grouped_df_comum
    for dataframe_list, grouped_list in zip([df_list_blue, df_list_comum], [list_grouped_df_blue, list_grouped_df_comum]):
        for df in dataframe_list:
            grouped_df = df.groupby('BLOCK_DATE')
            grouped_list.append(grouped_df)
    
    return list_grouped_df_blue, list_grouped_df_comum


def criar_sumario_dataframes(list_grouped_df_comum, list_grouped_df_blue):
        
    ts_nft_comum = []  # Lista para armazenar os DataFrames criados a partir de list_grouped_df_comum
    ts_nft_blue = []   # Lista para armazenar os DataFrames criados a partir de list_grouped_df_blue

    # Loop para iterar sobre list_grouped_df_comum e list_grouped_df_blue
# Loop para iterar sobre list_grouped_df_comum e list_grouped_df_blue
    for label, grouped_list, ts_nft_list in [("Comum", list_grouped_df_comum, ts_nft_comum), ("Blue Chips", list_grouped_df_blue, ts_nft_blue)]:
        adicionar_mensagem(f"Elaboração das Série(s) Temporal(is) Multivariada(s) - NFT {label}")
    
        for i, ts in enumerate(grouped_list):
            # adicionar_mensagem(f"\ngrouped_df_{label} {i+1}")
        
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



def dividir_dataframes(ts_nft_blue):
    train_data_blue = []
    validation_data_blue = []
    test_data_blue = []

    for df in ts_nft_blue:
        # Dividir o DataFrame em treinamento (60%), validação (20%) e teste (20%)
        train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
        validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Adicionar os DataFrames resultantes às listas correspondentes
        train_data_blue.append(train_df)
        validation_data_blue.append(validation_df)
        test_data_blue.append(test_df)
    
    return train_data_blue, validation_data_blue, test_data_blue
    
    
def calculate_metrics(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return accuracy, precision, recall, f1

def print_confusion_matrix(true_labels, predictions):
    confusion = confusion_matrix(true_labels, predictions)
    adicionar_mensagem("Matriz de Confusão:")
    adicionar_mensagem(confusion)
    
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


def print_confusion_matrices(confusion_matrix_list, phase):
    for i, cm in enumerate(confusion_matrix_list):
        cm_str = np.array2string(cm)  # Converte a matriz de confusão para uma string
        adicionar_mensagem(f"Matriz de Confusão ({phase} {i+1}):")
        adicionar_mensagem(cm_str)
        

def encontrar_tp(confusion_matrix_val):
    tp_found = confusion_matrix_val[0, 0] > 0
    return tp_found

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
    adicionar_mensagem("\nMédia dos Indicadores (Validação):\n")
    print_metrics(validation_metrics_mean)

    # Imprimir matrizes de confusão para validação
    # print_confusion_matrices(validation_confusion_matrix_list, "Validação")

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
    adicionar_mensagem("\nMédia dos Indicadores (Teste):\n")
    print_metrics(test_metrics_mean)


def avaliar_modelo_lof(train_data_blue, ts_nft_comum):
    lof = LocalOutlierFactor(novelty=True, n_neighbors=20, contamination=0.3)
    
    # Listas para armazenar métricas e resultados
    accuracy_scores, precision_scores, recall_scores, f1_scores, true_positive_anomalies = [], [], [], [], []
    accuracy_scores_heatmap, precision_scores_heatmap, recall_scores_heatmap, f1_scores_heatmap = [], [], [], []

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
        adicionar_mensagem(f"\nMétricas:\n")
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
                adicionar_mensagem("\nO conjunto de dados possui registro(s) classificados como anomalia(s) e possui recall superior ao limiar definido de " + format(limiar_recall *100, '.2f') + "%. \nRecall: " + format(average_recall *100, '.4f') +"%.")
                modelo1 = 1
            else:
                adicionar_mensagem("\nApesar de possuir registro(s) classificado(s) como anomalia(s) o conjunto de dados não será classificado como anômalo, pois possui um recall abaixo do limiar definido de " + format(limiar_recall *100, '.2f') + "%.\nRecall: " + format(average_recall *100, '.4f') +"%.")
                modelo1 = 0
        else:
            adicionar_mensagem("\nO conjunto de dados não é classificado como anômalo, pois não há nenhum registro classificado pelo modelo. \nRecall: " + format(average_recall *100, '.4f') +"%.")
            modelo1 = 0

    return modelo1


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
    adicionar_mensagem("\nMédia dos Indicadores (Validação):\n")
    print_metrics(validation_metrics_mean)

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
    adicionar_mensagem("\nMédia dos Indicadores (Teste):\n")
    print_metrics(test_metrics_mean)

    
def avaliar_modelo_isolation_forest(train_data_blue, ts_nft_comum):
    isolation_forest_model = IsolationForest(contamination='auto', random_state=42)
    
    # Listas para armazenar métricas e resultados
    accuracy_scores, precision_scores, recall_scores, f1_scores, true_positive_anomalies = [], [], [], [], []
    accuracy_scores_heatmap, precision_scores_heatmap, recall_scores_heatmap, f1_scores_heatmap = [], [], [], []
    
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
        adicionar_mensagem(f"\nMétricas:\n")
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
                adicionar_mensagem("\nO conjunto de dados possui registro(s) classificados como anomalia(s) e possui recall superior ao limiar definido de " + format(limiar_recall *100, '.2f') + "%. \nRecall: " + format(average_recall *100, '.4f') +"%.")
                modelo2 = 1
            else:
                adicionar_mensagem("\nApesar de possuir registro(s) classificado(s) como anomalia(s) o conjunto de dados não será classificado como anômalo, pois possui um recall abaixo do limiar definido de " + format(limiar_recall *100, '.2f') + "%.\nRecall: " + format(average_recall *100, '.4f') +"%.")
                modelo2 = 0
        else:
            adicionar_mensagem("\nO conjunto de dados não é classificado como anômalo, pois não há nenhum registro classificado pelo modelo. \nRecall: " + format(average_recall *100, '.4f') +"%.")
            modelo2 = 0

    return modelo2


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

    # Imprimir a média dos indicadores
    adicionar_mensagem("\nMédia dos Indicadores (Todas as Validações):\n")
    adicionar_mensagem("Acurácia: " + format(average_accuracy, '.4f'))
    adicionar_mensagem("Precisão: " + format(average_precision, '.4f'))
    adicionar_mensagem("Recall: " + format(average_recall, '.4f'))
    adicionar_mensagem("F1-score: " + format(average_f1, '.4f'))

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

    # Imprimir a média dos indicadores
    adicionar_mensagem("\nMédia dos Indicadores (Todos os Testes):\n")
    adicionar_mensagem("Acurácia: " + format(average_accuracy, '.4f'))
    adicionar_mensagem("Precisão: " + format(average_precision, '.4f'))
    adicionar_mensagem("Recall: " + format(average_recall, '.4f'))
    adicionar_mensagem("F1-score: " + format(average_f1, '.4f'))

    for lista in (accuracy_scores, precision_scores, recall_scores, f1_scores):
        while lista:
            lista.pop()    


            
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
        adicionar_mensagem("\nMatriz de Confusão para o conjunto de Teste " + str(i + 1) + ":")
        ct_str = np.array2string(confusion_test)  # Converte a matriz de confusão para uma string
        adicionar_mensagem(ct_str)

        # Imprimir as métricas para o conjunto de teste
        adicionar_mensagem("\nMétricas para o conjunto de Teste " + str(i + 1) + ":\n")
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
                adicionar_mensagem("\nO conjunto de dados possui registro(s) classificados como anomalia(s) e possui recall superior ao limiar definido de " + format(limiar_recall *100, '.2f') + "%. \nRecall: " + format(average_recall *100, '.4f') +"%.")
                modelo3 = 1
            else:
                adicionar_mensagem("\nApesar de possuir registro(s) classificado(s) como anomalia(s) o conjunto de dados não será classificado como anômalo, pois possui um recall abaixo do limiar definido de " + format(limiar_recall *100, '.2f') + "%.\nRecall: " + format(average_recall *100, '.4f') +"%.")
                modelo3 = 0
        else:
            adicionar_mensagem("\nO conjunto de dados não é classificado como anômalo, pois não há nenhum registro classificado pelo modelo. \nRecall: " + format(average_recall *100, '.4f') +"%.")
            modelo3 = 0

    return modelo3


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


# Inicialização da GUI
root = tk.Tk()
root.title("Processo Integrado de Consultas de NFTs")
root.geometry("600x600")

# StringVar com rastreador
texto_entrada = tk.StringVar()
texto_entrada.trace("w", limitar_tamanho)

instrucoes_label = tk.Label(root, text="Digite o código hash da NFT:")
instrucoes_label.pack(pady=10)

# Campo de entrada com limite de caracteres
entrada_hash = tk.Entry(root, textvariable=texto_entrada, width=50)
entrada_hash.pack(pady=10)

frame_botoes = tk.Frame(root)
frame_botoes.pack(pady=10)

botao_consultar = tk.Button(frame_botoes, text="Consultar", command=consultar_nft)
botao_consultar.pack(side=tk.LEFT, padx=5)

botao_limpar = tk.Button(frame_botoes, text="Limpar", command=limpar_entrada)
botao_limpar.pack(side=tk.LEFT, padx=5)

# Adicionando uma área de texto para mensagens
text_area = scrolledtext.ScrolledText(root, height=10, state=tk.DISABLED)
text_area.pack(pady=10)

# Autenticação BigQuery
client_bigquery = autenticar_bigquery()

# Ler Chave API Etherscan
api_key_etherscan = ler_chave_etherscan()

# Carregar os arquivos CSV das NFTs Blue Chips
list_df_blue = []
list_df_blue = lista_df()

executar_consulta_bigquery


# Carregar arquivo CSV da NFT Comum
list_df_comum = []
df_comum = carregar_arquivo_csv()
if df_comum is not None:
    list_df_comum.append(df_comum)


# Processando a lista de DataFrames list_df_blue
adicionar_mensagem(f"\nPré-Processamento das Informações de NFTs Blue Chips.")
adicionar_mensagem(f"Pré-Processamento das Informações de NFT Comum.")

df_list_blue = []
for df_blue in list_df_blue:
    process_dataframe(df_blue)
    df_list_blue.append(df_blue)

# Processando a lista de DataFrames list_df_comum
df_list_comum = []
for df_comum in list_df_comum:
    process_dataframe(df_comum)
    df_list_comum.append(df_comum)

    
# Agrupamento dos dados utilizando a coluna BLOCK_DATE
adicionar_mensagem(f"\nAgrupamento das Informações de NFTs Blue Chips.")
adicionar_mensagem(f"Agrupamento das Informações de NFT Comum.\n")

list_grouped_df_blue, list_grouped_df_comum = [], []

# Chamada da função agrupar_dataframes_por_data
list_grouped_df_blue, list_grouped_df_comum = agrupar_dataframes_por_data(df_list_blue, df_list_comum)


# Criação das Series Temporais Multivariadas
ts_nft_comum, ts_nft_blue = [], []  

# Chamada da função criar_sumario_dataframes
ts_nft_comum, ts_nft_blue = criar_sumario_dataframes(list_grouped_df_comum, list_grouped_df_blue)


# Normalização das Séries Temporais Multivariadas ts_nft_blue
normalize_dataframes(ts_nft_blue)
adicionar_mensagem(f"\nNormalização das Séries Temporais Multivariadas de NFTs Blue Chips.")

# Normalização da Série Temporal Multivariada ts_nft_comum
normalize_dataframes(ts_nft_comum)
adicionar_mensagem(f"Normalização da Série Temporal Multivariada de NFT Comum normalizada.")


# Dividir cada DataFrame em ts_nft_blue em conjuntos de treinamento, validação e teste
train_data_blue = []
validation_data_blue = []
test_data_blue = []

# Chamada da função divisão dos dados de NFTs Blue Chips para treinamento, validação e teste
adicionar_mensagem(f"\nDivisão das NFTs Blue Chips - Treinamento, Validação e Teste.")
train_data_blue, validation_data_blue, test_data_blue = dividir_dataframes(ts_nft_blue)


# LOF
adicionar_mensagem(f"\nUtilização modelo de AM - LOF.")

adicionar_mensagem(f"\nTreino, Validação e Testes do modelo LOF - NFTs Blue Chips.")
treinar_validar_testar_lof(train_data_blue, validation_data_blue, test_data_blue)

adicionar_mensagem(f"\nTestes do modelo LOF - NFT Comum.")
modelo1 = avaliar_modelo_lof(train_data_blue, ts_nft_comum)


# Isolation Forest
adicionar_mensagem(f"\nUtilização modelo de AM - Isolation Forest.")

adicionar_mensagem(f"\nTreino, Validação e Testes do modelo Isolation Forest - NFTs Blue Chips.")
treinar_validar_testar_isolation_forest(train_data_blue, validation_data_blue, test_data_blue)

adicionar_mensagem(f"\nTestes do modelo Isolation Forest - NFT Comum.")
modelo2 = avaliar_modelo_isolation_forest(train_data_blue, ts_nft_comum)


# DBScan
adicionar_mensagem(f"\nUtilização modelo de AM - DBScan.")

adicionar_mensagem(f"\nTreino, Validação e Testes do modelo DBScan - NFTs Blue Chips.")
treinar_validar_testar_dbscan(train_data_blue, validation_data_blue, test_data_blue)

adicionar_mensagem(f"\nTestes do modelo DBScan - NFT Comum.")
modelo3 = avaliar_modelo_dbscan(ts_nft_comum)

#Comite de Classificação
adicionar_mensagem(f"\nComitê de Classificação de NFT.")
resultado = decidir_anomalia(modelo1, modelo2, modelo3)
adicionar_mensagem(f"\n{resultado}")


root.mainloop()
