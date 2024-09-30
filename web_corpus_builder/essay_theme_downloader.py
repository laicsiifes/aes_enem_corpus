from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import re
import json
import config


def scroll_to_bottom( driver:str ):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")


def click_load_more_button( driver:str ):
    load_more_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "load-more-btn")))
    load_more_button.click()


# Salvar arquivo com todos os temas encontrados
def salve_theme_json( url:str=None , limit_scroll:int=None , output_json_themes:str=None):
    '''
        Salvar JSON com temas
    '''
    url = url if url is not None else config.URL_THEMES
    limit_scroll = limit_scroll if limit_scroll is not None else config.LIMIT_SCROLL
    output_json_themes = output_json_themes if output_json_themes is not None else config.JSON_THEMES

    chrome = webdriver.Chrome()
    chrome.get(url)
    time.sleep(3)

    for _ in range(limit_scroll):
        # Rola a página para o final
        scroll_to_bottom(chrome)
        print("Aguarde scroll...")      
        time.sleep(1)
        try:
            click_load_more_button(chrome)
        except Exception as e:
            print("Erro ao esperar pelo desaparecimento do iframe:", e)
    
    # Pegar o HTML da página depois do carregamento
    print("Pegando HTML após carregamento...") 

    page_html = chrome.page_source
    soup = BeautifulSoup(page_html, 'html.parser')
    tables = soup.find_all('table', id='table-temas')
    count_themes = 0
    json_list = []
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            theme_dic = {}
            cells = row.find_all('td')
            if len(cells) != 0:
                theme = cells[0].find('a').get_text(strip=True)
                link = cells[0].find('a', href=True)
                url_return = re.findall(r"redacoes\/(.*htm)",link['href'])
                url_full = config.URL_THEMES + "/" + url_return[0]
                mes_ano = cells[1].get_text(strip=True).split("/")
                theme_dic = {
                    "tema" : theme if theme else "" ,
                    "mes"  : mes_ano[0] if mes_ano else "",
                    "ano"  : mes_ano[1] if mes_ano else "",
                    "url"  : url_full if url_full != [] else ""
                }
                json_list.append(theme_dic)
                count_themes += 1
                #print(theme_dic)

    # Salvar JSON com os temas
    with open(output_json_themes, "w",encoding="utf-8") as file:
        json.dump(json_list, file, ensure_ascii=False, indent=4)

    print("Total de temas encontrados: ", count_themes)