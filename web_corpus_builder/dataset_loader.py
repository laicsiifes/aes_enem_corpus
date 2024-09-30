import json
import os
import pandas as pd
from tqdm import tqdm
import config

def load_dataset_into_dataframe( dir:str=None ) -> pd.DataFrame:
    try:
        dir = dir or config.OUTPUT_FOLDER
        
        df = pd.DataFrame(columns=['c1', 'c2', 'c3', 'c4', 'c5', 'essay'])
        files = os.listdir(dir)
        with tqdm(total=len(files), desc='Carregando...') as fbar:
            for file in files:        
                json_file = os.path.join(dir,file)
                with open(json_file, 'r', encoding='utf-8' ) as f:
                    data = json.load(f)
                    essays = data[0]['lista_de_redacoes']
                    ano = data[0]['ano']
                    mes = data[0]['mes']
                    
                    for essay in essays:
                        if essay != None and len(essay['notas']) == 5:
                            df = df._append({
                            'c1': essay['notas'][0],
                            'c2': essay['notas'][1],
                            'c3': essay['notas'][2],
                            'c4': essay['notas'][3], 
                            'c5': essay['notas'][4],
                            'marcacoes': essay['marcacoes'],
                            'id_ref': essay['id'],
                            'essay': "\n".join(essay['lista_de_paragrafos']),
                            'ano': ano,
                            'mes': mes,
                            'total_paragrafos': len(essay['lista_de_paragrafos']),
                            'lista_de_paragrafos': essay['lista_de_paragrafos'],
                            'notas_originais': essay['notas_originais']
                            }, ignore_index=True)
                fbar.update(1)  
        return df
    except Exception as e:
        print(f'Falha no carregamento do JSON. Erro: {e}')


def load_dataset_into_csv( dir:str=None ):
    try:
        dir = dir or config.OUTPUT_FOLDER
        
        df = pd.DataFrame(columns=['c1', 'c2', 'c3', 'c4', 'c5', 'essay'])
        files = os.listdir(dir)
        with tqdm(total=len(files), desc='Carregando...') as fbar:
            for file in files:        
                json_file = os.path.join(dir,file)
                with open(json_file, 'r', encoding='utf-8' ) as f:
                    data = json.load(f)
                    essays = data[0]['lista_de_redacoes']
                    ano = data[0]['ano']
                    mes = data[0]['mes']
                    
                    for essay in essays:
                        if essay != None and len(essay['notas']) == 5:
                            df = df._append({
                            'c1': essay['notas'][0],
                            'c2': essay['notas'][1],
                            'c3': essay['notas'][2],
                            'c4': essay['notas'][3], 
                            'c5': essay['notas'][4],
                            'marcacoes': essay['marcacoes'],
                            'id_ref': essay['id'],
                            'essay': "\n".join(essay['lista_de_paragrafos']),
                            'ano': ano,
                            'mes': mes,
                            'total_paragrafos': len(essay['lista_de_paragrafos']),
                            'lista_de_paragrafos': essay['lista_de_paragrafos'],
                            'notas_originais': essay['notas_originais']
                            }, ignore_index=True)
                fbar.update(1)  

        # Salva/Converte Dataframe em CSV de saída
        df.to_csv('dataset.csv', index=False)  # index=False para não incluir o índice no CSV
    except Exception as e:
        print(f'Falha no carregamento do JSON. Erro: {e}')