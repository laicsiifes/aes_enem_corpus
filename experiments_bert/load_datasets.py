import pandas as pd
import spacy
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, DatasetDict
import os, json
import re
from tqdm import tqdm
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings('ignore')

PATH_CORPUS_PROPOSTO = "datasets/our_dataset"
PATH_CORPUS_ESSAY_BR = "datasets/extended_essay_br/extended_essay-br.csv"
PROMPT_PATH = "datasets/extended_essay_br/prompts.csv"


def read_proposed_corpus(corpus_path: str) -> pd.DataFrame:
    try:
        df = pd.DataFrame(columns=['id','essay','c1', 'c2', 'c3', 'c4', 'c5'])
        files = os.listdir(corpus_path)
        with tqdm(total=len(files), desc='Carregando...') as fbar:
            for file in files:        
                json_file = os.path.join(corpus_path, file)
                with open(json_file, 'r', encoding='utf-8' ) as f:
                    data = json.load(f)
                    essays = data[0]['lista_de_redacoes']
                    ano = data[0]['ano']
                    mes = data[0]['mes']
                    for essay in essays:
                        if essay != None and len(essay['notas']) == 5:
                            essay_text = "\n".join(essay['lista_de_paragrafos'])
                            df = df._append({
                                'id': essay['id'],
                                'essay': essay_text,
                                'c1': essay['notas'][0],
                                'c2': essay['notas'][1],
                                'c3': essay['notas'][2],
                                'c4': essay['notas'][3],
                                'c5': essay['notas'][4]
                            }, ignore_index=True)
                fbar.update(1)  
        return df
    except Exception as e:
        print(f'Falha no carregamento do JSON. Erro: {e}')


def read_extend_essay_br(corpus_path: str, prompt_path: str = None) -> pd.DataFrame:
    essays_df = pd.read_csv(corpus_path, converters={'essay': eval})
    prompts_dict = None
    if prompt_path is not None:
        prompts_df = pd.read_csv(prompt_path)
        ids = prompts_df['id'].tolist()
        prompts_descs = prompts_df['description'].tolist()
        new_prompts = []
        for prompt in prompts_descs:
            if prompt.strip().startswith("["):
                list_texts = list(map(str, prompt[1:-1].split(',')))
                prompt = ' '.join(list_texts).strip()
            new_prompts.append(prompt)
        prompts_dict = {id_: desc for id_, desc in zip(ids, new_prompts)}
    essays = []
    set_normalized_essays = set()
    
    for i, rows in essays_df.iterrows():
        prompt_id = rows['prompt']
        motivating_situation = None
        if prompts_dict is not None:
            motivating_situation = prompts_dict[prompt_id]
        essay_text = '\n'.join(rows['essay']).strip()
        if len(essay_text.split()) > 0:
            essay_text_normalized = essay_text.lower().replace(' ', '').strip()
            essay_text_normalized = re.sub(r'[^\w\s]', '', essay_text_normalized)

            if essay_text_normalized not in set_normalized_essays:
                essay = {
                'id': int(i)+1,
                'essay': essay_text,
                'c1': int(rows['c1']),
                'c2': int(rows['c2']),
                'c3': int(rows['c3']),
                'c4': int(rows['c4']),
                'c5': int(rows['c5'])
                }
                essays.append(essay)            
                set_normalized_essays.add(essay_text_normalized)

    df_out = pd.DataFrame(essays)
    return df_out


def read_dataset_sourceAWithGraders() -> pd.DataFrame:

    dataset2A = load_dataset("kamel-usp/aes_enem_dataset", "sourceAWithGraders", cache_dir="/tmp/aes_enem")
    train_dataset = dataset2A['train']
    validation_dataset = dataset2A['validation']
    test_dataset = dataset2A['test']

    df1 = pd.DataFrame(train_dataset)
    df2 = pd.DataFrame(validation_dataset)
    df3 = pd.DataFrame(test_dataset)
    concat_df = pd.concat([df1, df2, df3])

    grades_df = pd.DataFrame(concat_df['grades'].tolist(), index=concat_df.index)
    grades_df.columns = [f'c{i+1}' for i in range(grades_df.shape[1])]
   
    grades_df = grades_df.astype(int)

    df_expanded = pd.concat([concat_df.drop(columns=['grades']), grades_df], axis=1)
    df_reduced = df_expanded[['id', 'essay_text', 'c1', 'c2', 'c3', 'c4', 'c5']]
    df_reduced = df_reduced.rename(columns={'essay_text': 'essay'})

    return df_reduced


def read_dataset_sourceB() -> pd.DataFrame:
    
    dataset2B = load_dataset("kamel-usp/aes_enem_dataset", "sourceB", cache_dir="/tmp/aes_enem")
    df= pd.DataFrame(dataset2B['full'])

    grades_df = pd.DataFrame(df['grades'].tolist(), index=df.index)
    grades_df.columns = [f'c{i+1}' for i in range(grades_df.shape[1])]

    df_expanded = pd.concat([df.drop(columns=['grades']), grades_df], axis=1)
    df_reduced = df_expanded[['id', 'essay_text','c1', 'c2', 'c3','c4','c5']]
    df_reduced = df_reduced.rename(columns={'essay_text': 'essay'})

    return df_reduced


# Versão com usando similaridade entre os textos
def filter_common_essays(essays_a, essays_b):

    texts = pd.concat([essays_a['essay'], essays_b['essay']], ignore_index=True)

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(texts)

    tfidf_A = tfidf_matrix[:len(essays_a)]
    tfidf_B = tfidf_matrix[len(essays_a):]

    similaridade = cosine_similarity(tfidf_A, tfidf_B)

    similarity_threshold = 0.85

    unique_indices_of_B = []

    for i, sim in enumerate(similaridade.T):
        if max(sim) < similarity_threshold:
            unique_indices_of_B.append(i)

    # Filtrar os elementos únicos de B
    unique_df_B = essays_b.iloc[unique_indices_of_B]

    return unique_df_B







