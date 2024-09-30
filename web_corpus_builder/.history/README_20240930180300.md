# Web Scraping - AES ENEM

#### Descrição:
-Web Scraping para coletar redações do site Brasil Escola e gerar um dataset.


# Instruções de Setup

Siga os passos abaixo para configurar o ambiente e executar o projeto.


## Passo 1: Clonar o Repositório

```bash
git clone https://github.com/laicsiifes/aes_enem_corpus.git
cd aes_enem_corpus
```

## Passo 2: Criar e Ativar o Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
.\venv\Scripts\activate  # Windows

```


## Passo 3: Instalar Depedências

```bash
pip install -r requirements.txt

```

## Passo 4: Ajuste as configurações no arquivo 'config.py'

```bash
# Defina o período
START_MONTH = 6
END_MONTH = 9
START_YEAR = 2024
END_YEAR = 2024

```

## Passo 5: Executar o Projeto

```bash
python web_corpus_main.py

```