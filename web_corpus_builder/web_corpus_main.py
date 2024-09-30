from essay_theme_downloader import salve_theme_json
from essay_processing_manager import salve_list_of_essays_json
from dataset_loader import load_dataset_into_csv
import pandas as pd
import time


if __name__ == "__main__":
    
    inicio = time.time()

    #### - Carregar e baixar os temas das redações
    print("###############################################################################################################################")
    print("Carregando tabela de temas....")
    print("###############################################################################################################################")    

    salve_theme_json()
    time.sleep(3)

    #### - Processar e salvar redações
    print("###############################################################################################################################")
    print("#  Iniciando processamento...")
    print("###############################################################################################################################")
    print("\n")
    inicio = time.time()

    salve_list_of_essays_json()

    #### - Carregando dados para CSV
    print("###############################################################################################################################")
    print("#  Carregando dados para CSV...")
    print("###############################################################################################################################")
    print("\n")

    load_dataset_into_csv()

    print("\n\nExibir CSV")
    # Carregar o arquivo CSV em um DataFrame
    df = pd.read_csv('dataset.csv')

    # Exibir o DataFrame
    print(df)

    #### - Finalizando...
    print("###############################################################################################################################")
    print("Processamento finalizado!")
    print("###############################################################################################################################")

    fim = time.time()
    tempo_execucao = fim - inicio

    print("Tempo de processamento: ", tempo_execucao, "segundos")





    
