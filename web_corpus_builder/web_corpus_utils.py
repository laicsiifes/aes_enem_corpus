import requests


def get_html( url:str ):
    html = ""
    response = requests.get(url)
    print('STATUS CODE: ', response.status_code)
    print("URL:", url)
    if response.status_code == 200:
        response.encoding = 'utf-8'
        html = response.text
    return html


def salve_log_to_file(log, nome_arquivo):
    with open(nome_arquivo, 'a', encoding='utf-8') as arquivo:
        arquivo.write(log + '\n')



