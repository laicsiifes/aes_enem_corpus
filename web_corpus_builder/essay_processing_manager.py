from web_corpus_utils import get_html, salve_log_to_file
from bs4 import BeautifulSoup
from essay_data_extractor import load_corrections
import json
import config
import os

def get_motivation_text( html_text:str ):

    try: 
        soup = BeautifulSoup(html_text, 'html.parser')
        motivating_text = soup.find_all("p")[0].get_text(strip=True)
        return motivating_text
    except Exception as e:
        print(f'Erro: {e}.')
        salve_log_to_file(str(e), config.LOG)
        return None


def get_urls_list(html_text:str):
    urls_list = []
    try:
        # Acessar tabela com a lista
        soup = BeautifulSoup(html_text, 'html.parser')    
        table = soup.find('table', id='redacoes_corrigidas')
       
        urls = table.find_all('a')
        for url in urls:
            urls_list.append(url['href'])
        return urls_list
    
    except Exception as e:
        print("Erro ao gerar lista de URL.")        
        print(f'Erro: {e}.')
        salve_log_to_file(str(e), config.LOG)
        return None
    

def load_list_of_essays(theme_url: str):
    motivating_text = ""
    list_of_eassays  = []
    
    # Acessar a url do tema e pegar o HTML
    html = get_html(theme_url)

    # Pegar texto motivador no HTML
    motivating_text = get_motivation_text(html)

    # Carregar urls das redacoes em uma lista da tabela do HTML
    urls_list = get_urls_list(html)

    # Para cada url de redacao carregar as correcoes e montar dic e list de saída
    for url in urls_list:
        correction = load_corrections(url)
        list_of_eassays.append(correction)

    return motivating_text, list_of_eassays

def map_month(month:str):
    try:
        list_month = [
            'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
            'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
        ]
        return list_month.index(month) + 1
    except:
        return month

def salve_list_of_essays_json( 
        json_themes: str=None,
        start_month: str=None,
        end_month: str=None,
        start_year:int=None, 
        end_year:int=None ):

    try:
        json_themes = json_themes if json_themes is not None else config.JSON_THEMES
        start_month = start_year if start_month is not None else config.START_MONTH
        end_month = end_month if end_month is not None else config.END_MONTH
        start_year = start_year if start_year is not None else config.START_YEAR
        end_year = end_year if end_year is not None else config.END_YEAR
    
        total = 0
        # Acessar JSON com os temas 
        with open(json_themes, 'r', encoding='utf-8') as f:
            list_of_themes = json.load(f)
        
        
        # Acessar URL de cada para pegar a lista de redações de cada tema
        for theme in list_of_themes:
            
            print("Ano: {:5} - Mês: {:10} - Tema: {:}".format( theme['ano'], theme['mes'],theme['tema']) )
            mes = map_month(theme['mes'])

            if (
                int(theme['ano']) >= start_year and 
                int(theme['ano']) <= end_year and
                mes >= start_month and 
                mes <= end_month
                ):

                json_list = []        
                
                # Carregar lista de redações (com as correções) e texto motivador 
                motivating_text, list_with_eassays = load_list_of_essays(theme['url'])                

                theme_dic = {
                    "ano": theme['ano'],
                    "mes": theme['mes'],
                    "url": theme['url'],
                    "tema": theme['tema'],
                    "texto_motivador": str(motivating_text) if motivating_text else "",
                    "lista_de_redacoes": list_with_eassays if list_with_eassays else []
                }

                print("Total de redações do tema:", len(list_with_eassays))
                total += len(list_with_eassays)
                json_list.append(theme_dic)                  

                # Salvar o JSON final do tema  
                mes = map_month(theme['mes'])
                file_name = str(theme['ano']) + '_' + str(mes) + '.json'
                output_file = os.path.join(config.OUTPUT_FOLDER, file_name)

                with open(output_file, "w",encoding="utf-8") as file:
                    json.dump(json_list, file, ensure_ascii=False, indent=4)
                
                print("\n")            
            
        print("Total de redações coletadas: ", total)

    except Exception as e:
        erro = f'Erro: {e}'
        salve_log_to_file(erro, config.LOG)
        return None
