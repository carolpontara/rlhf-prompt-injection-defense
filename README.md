
# 🛡️ LLM Security: Detecção e Mitigação de Prompt Injection Attacks com RLHF

Este repositório contém o código-fonte e os scripts utilizados no experimento de fine-tuning supervisionado com feedback humano para mitigar ataques de Prompt Injection em modelos de linguagem (LLMs), utilizando o modelo base GPT-2 e a biblioteca HuggingFace Transformers.

## 📚 Descrição do Projeto

Prompt Injection Attacks consistem em manipular o comportamento de modelos de linguagem através de comandos maliciosos inseridos no prompt. Este projeto explora uma abordagem bioinspirada de aprendizado por reforço supervisionado humano para identificar e mitigar essas injeções, simulando um processo de curadoria e refinamento iterativo.

O pipeline é dividido em três etapas principais:

1. **Geração de Respostas**: respostas são geradas automaticamente para prompts contendo ou não ataques.
2. **Fine-Tuning com Feedback Humano**: um dataset anotado com scores humanos é usado para refinar o modelo.
3. **Avaliação Comparativa**: são comparadas respostas do modelo original vs modelo refinado.

## 🛠️ Estrutura do Repositório

````
📁 llm-security-rlhf/
│
├── data/
│   ├── prompts.csv               # Prompts originais (com/sem injeções)
│   ├── generated\_responses.csv   # Respostas do modelo base
│   ├── evaluated\_responses.csv   # Respostas anotadas com score
│   └── model\_comparison.csv      # Comparativo final entre modelos
│
├── scripts/
│   ├── generate\_responses.py     # Gera respostas com GPT-2
│   ├── finetune\_model.py         # Fine-tuning supervisionado com score > 1
│   └── evaluate\_model.py         # Compara desempenho dos modelos
│
├── README.md
└── requirements.txt

````

## 🚀 Como Rodar o Projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/llm-security-rlhf.git
cd llm-security-rlhf
````

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

> Requisitos principais: `transformers`, `datasets`, `torch`, `pandas`, `scikit-learn`, `tqdm`

### 3. Execute os scripts

#### Gerar respostas

```bash
python scripts/generate_responses.py
```

#### Aplicar fine-tuning supervisionado

> Certifique-se de anotar o arquivo `generated_responses.csv` com colunas `score`, `prompt_traduzido`, `resposta`.

```bash
python scripts/finetune_model.py
```

#### Avaliar modelos (original vs. fine-tuned)

```bash
python scripts/evaluate_model.py
```

---

## 📈 Resultados Esperados

* Melhoria na qualidade e segurança das respostas do modelo refinado
* Redução da suscetibilidade a prompts maliciosos
* Dataset com anotações humanas úteis para estudos futuros sobre alinhamento de LLMs

---

## 💡 Tecnologias Utilizadas

* [HuggingFace Transformers](https://huggingface.co/transformers/)
* PyTorch
* Pandas
* Python 3.10+
* CUDA (opcional para aceleração com GPU)

---

## 🧠 Autor

Desenvolvido por \[Ana Caroline Silva Pontara] – Mestranda em Ciência da Computação na UNESP, membro do [LARS - Advanced Network and Security Lab](https://www.keltoncosta.com.br/LARS/index.html)

---

## 📜 Licença

Este projeto é livre para uso acadêmico. Para fins comerciais ou redistribuição, entre em contato com os autores.
