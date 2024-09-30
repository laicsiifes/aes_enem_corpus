from web_corpus_utils import get_html, salve_log_to_file
from bs4 import BeautifulSoup
import re
import config


def is_valid_text(tex_html):
    if tex_html == '':
        return False
    else:
        new_text = re.sub(r"\r|\n|\xa0\s|\<p/?>", "", tex_html)
        return False if new_text == '' else True


def load_paragraphs( soup:BeautifulSoup ):

    list_paragrafos = []
    # Pegar tags <p>
    paragrafos = soup.find_all('p')
 
    for p in paragrafos:
        # Tratar parágrafos com tag <br/>
        if "<br/>" in str(p):
            quebra = str(p).split("<br/>")
            valid_paragraphs = [br for br in quebra if is_valid_text(br)]
            for br_p in valid_paragraphs:
                paragrafo = re.sub(r"</?p>", "", str(br_p))
                list_paragrafos.append(paragrafo)

        else:
            paragrafo = re.sub(r"</?p>", "", str(p))
            list_paragrafos.append(paragrafo)
    return list_paragrafos


def load_appointments( html_text: str):
    try:
        soup = BeautifulSoup(html_text, 'html.parser')

        # text comentado
        text = soup.get_text()
        text = text.replace("\n", "").replace("\r", "").replace("\t", "")
        text = text.replace("\u200B", "").replace("\xa0"," ").strip()   
        text = re.sub(r"\s+", " ", text)

        # Texto com hml
        text_tag = str(soup)
        text_tag = text_tag.replace("\n", "").replace("\r", "").replace("\t", "")
        text_tag = text_tag.replace("\u200B", '').replace("\xa0"," ").strip()      
        text_tag = re.sub(r"\s+", ' ', text_tag)

        comments_groups = re.findall(r"<[strong|u][^>]*>.*?(\[.*?\]).*?<\/[strong|u]|<span[^>]*>.*?(\(.*?\)).*?<\/span>", text_tag)
        comments = [match[0] if match[0] else match[1] for match in comments_groups]

        posicao_novo_texto = 0
        posicao_texto_comentado = 0
        marcacoes = []
        new_text = ""
        matches = []
        spans_included = set()
        
        if comments:
            for comment in comments:
                
                results = list(re.finditer(re.escape(comment),text))

                for match in results:
                    if match.span() not in spans_included:

                        spans_included.add(match.span())

                        # Acumular as correspondências
                        matches.append(match)

            # Ordernar pela posição inicial para que não se repita em casos de comentários iguais
            sorted_matches = sorted(matches, key=lambda match: match.span()[0])
            
            # Buscar posição no texto
            for m in sorted_matches:
                
                start = int(m.start())
                end = int(m.end())  
                comment = text[start:end]
                       
                new_text += text[posicao_texto_comentado:start].lstrip()
                
                posicao_texto_comentado = end

                # Posição no novo texto
                posicao_novo_texto = len(new_text)

                marcacoes.append({
                            "anotacao": comment,
                            "posicao": posicao_novo_texto
                        })

                #print(comment,posicao_novo_texto)


        new_text += text[posicao_texto_comentado:].strip()

        #print(new_text)
        #print('--------------------------')
        
        return marcacoes, new_text

    
    except Exception as e:
        print(f'Erro ao carregar marcações de correção do texto. Erro: {e}')
        salve_log_to_file(str(e), config.LOG)
        return None
    

def load_notes(list_paragrafos: list):
    
    list_paragrafos_texto_original = []
    notes = []
    count= 0
    for paragrafo in list_paragrafos:
        #'\xa0'
        if str(paragrafo).strip() != '':
            count += 1
            #print("\nPARAGRAFO", count) 
            
            # Pegar as anotacoes
            anotacoes, novo_texto = load_appointments(str(paragrafo))
            if anotacoes != []:
                dic = {
                    "paragrafo": count,
                    "anotacoes": anotacoes

                }
                notes.append(dic)
            list_paragrafos_texto_original.append(novo_texto)
            anotacoes = []
            novo_texto = ""
    
    return notes, list_paragrafos_texto_original


def load_grades(html_text:str):
    
    try:
        
        grades = html_text.find_all("td", class_="simple-td")[3:]
        grades_map = {20:40,50:80,100:120,150:160}
        notas = []
        notas_originais = []

        for item in grades:
            nota = int(item.get_text(strip=True))

            if nota in grades_map:
                notas_originais.append(nota)
                nota = grades_map[nota]              
            notas.append(nota)     

        return notas, notas_originais
    except Exception as e:
        print("Erro ao carregar competencias.")
        print(f'Erro: {e}.')
        salve_log_to_file(e, config.LOG)
        return None


def get_date( soup:BeautifulSoup ):
    try:
        data_envio = soup.find("strong", string="Enviada em:").find_next_sibling(string=True).strip()
        return data_envio  
    except Exception as e:
        print("Erro ao carregar data de envio")
        print(f'Erro: {e}.')
        salve_log_to_file(e, config.LOG)
        return None


def get_title( soup:BeautifulSoup ):
    try:
        conteudo_titulo = soup.find("h1", class_="titulo-conteudo").get_text(strip=True)
        titulo = conteudo_titulo.split("- Banco de redações")[0].replace("\"","").strip()
        return titulo
    except Exception as e:
        print("Erro ao carregar título da redação.")
        print(f'Erro: {e}.')
        salve_log_to_file(str(e), config.LOG)
        return None


def load_corrections(url: str):

    try:
        # Acessar página da redação e pegar o HTML
        html = get_html(url)
        soup = BeautifulSoup(html, 'html.parser')

        # Pegar id
        id = re.findall("/(\d+)", url)
        print("ID", id[0] )

        # Pegar texto da redação com html
        area_redacao = soup.find("div", class_="area-redacao-corrigida")    
       
        # Pegar a lista de paragrafos 
        list_paragrafos = load_paragraphs(area_redacao)
        
        # Pegar anotações do texto - CORRIGIR o Soup pra pegar texto do paragrafo e nao do html
        marcacoes, list_paragrafos_texto_original = load_notes(list_paragrafos)

         # Pegar feedback_revisor
        area_comentario = soup.find('div', style='background-color:#f4f4f4; padding:10px; border:solid 1px #cecece')
        comentario = area_comentario.get_text( strip=True )

        # Pegar notas por competencias avaliadas
        area_competencias = soup.find("table", id="redacoes_corrigidas")
        notas, notas_originais = load_grades(area_competencias)

        # Pegar nota_final
        nota_final = sum(notas)

        # Pegar data de envio
        data_envio = get_date(soup)
   
        # Pegar título
        titulo = get_title(soup)
        
        # Pegar correcao
        dic_correcao =  {
            "id": str(id[0]) if id != [] else None,
            "url": url,
            "titulo": titulo if titulo else None,
            "data_envio": data_envio if data_envio else None,
            "lista_de_paragrafos": list_paragrafos_texto_original if list_paragrafos != [] else None,
            "marcacoes": list(marcacoes) if marcacoes else None,
            "feedback_revisor":str(comentario) if comentario else None,
            "notas_originais": list(notas_originais) if notas else None,
            "notas": list(notas) if notas else None,           
            "nota_final": nota_final if nota_final else None
        }
        return dic_correcao
    except Exception as e:
        print(f'Erro ao carregar correção. URL: {url}')
        print(f'Erro: {e}')

        #temp
        salve_log_to_file(url, config.LOG)
        salve_log_to_file(str(e), config.LOG)
        return None




if __name__ == "__main__":

    import pprint

    # 2017
    #url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/12552"

    # problema na marcação - o comentário está dividido em 2 spans e quebra o padrão - não tem como ajustar pq o html tem erro de tag
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/12554" 

    # iserindo paragrafo vazio
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/13282
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/12424"

    # Comentário com colchete e tag strong 
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/12339"
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/12357"

    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/12853"
    

    # ------------------------------------------------------------------------------------------------------------------------------------
    #2018
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/13797"

    # ------------------------------------------------------------------------------------------------------------------------------------
    # 2019
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/14397"

    # problema de parágrafos - o html está com as marcações erradas
    #url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/14978" 

    # ------------------------------------------------------------------------------------------------------------------------------------
    # 2020
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/15998"

    # ------------------------------------------------------------------------------------------------------------------------------------
    # 2021
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/17547"

    # ------------------------------------------------------------------------------------------------------------------------------------
    # 2022
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/18967"

    # ------------------------------------------------------------------------------------------------------------------------------------
    # 2023
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/20467"

    # ------------------------------------------------------------------------------------------------------------------------------------
    # 2024
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/21133"
    # url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/21380"
    url = "https://vestibular.brasilescola.uol.com.br/banco-de-redacoes/21522"



    correcoes = load_corrections(url)

    pprint.pprint(correcoes)
