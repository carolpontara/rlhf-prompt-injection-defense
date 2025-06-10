from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd
import torch
import os
from tqdm import tqdm

class ModelFinetuner:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_dir = "arquivos"
        os.makedirs(self.data_dir, exist_ok=True)

    def _clean_data(self, text):
        """Limpeza básica do texto"""
        if not isinstance(text, str):
            return ""
        return text.strip()

    def prepare_data(self, df):
        """Prepara dataset para treino com tratamento robusto"""
        try:
            # Verifica colunas disponíveis
            required_columns = {'prompt_traduzido', 'resposta', 'score'}  # Nomes das colunas no seu CSV
            available_columns = set(df.columns)
            
            # Verifica se todas as colunas necessárias existem
            if not required_columns.issubset(available_columns):
                missing = required_columns - available_columns
                raise ValueError(f"Colunas faltando no DataFrame: {missing}")
            
            # Renomeia colunas para nomes consistentes
            df = df.rename(columns={
                'prompt_traduzido': 'prompt',
                'resposta': 'response'
            }).copy()
            
            # Limpeza dos dados
            df["prompt"] = df["prompt"].apply(self._clean_data)
            df["response"] = df["response"].apply(self._clean_data)
            
            # Filtra linhas vazias e com score baixo
            df = df[(df["prompt"].str.len() > 0) & 
                   (df["response"].str.len() > 0) &
                   (df["score"] > 1)]  # Ajuste este threshold conforme necessário
            
            if len(df) == 0:
                raise ValueError("Nenhum dado válido para treinamento após filtragem")
            
            print(f"📊 Exemplos válidos para treino: {len(df)}")
            
            # Cria dataset
            dataset = Dataset.from_pandas(df)
            
            # Carrega tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Tokenização
            def tokenize_fn(examples):
                texts = [p + " " + r for p, r in zip(examples["prompt"], examples["response"])]
                return tokenizer(
                    texts,
                    truncation=True,
                    max_length=256,
                    padding="max_length",
                    return_tensors="pt"
                )
            
            return dataset.map(tokenize_fn, batched=True, batch_size=8), tokenizer

        except Exception as e:
            print(f"❌ Erro na preparação dos dados: {str(e)}")
            print("Colunas disponíveis no DataFrame:", df.columns.tolist())
            raise

    def finetune(self, train_data, tokenizer, output_dir="model_finetuned"):
        """Executa fine-tuning com tratamento de erros"""
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=2,
                num_train_epochs=3,
                save_steps=200,
                logging_steps=50,
                learning_rate=3e-5,
                weight_decay=0.01,
                gradient_accumulation_steps=2,
                fp16=torch.cuda.is_available(),
                save_total_limit=2
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                data_collator=data_collator
            )
            
            print("🚀 Iniciando fine-tuning...")
            trainer.train()
            
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            print(f"\n✅ Fine-tuning concluído! Modelo salvo em: {output_dir}")
            
        except Exception as e:
            print(f"❌ Erro durante o fine-tuning: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Carrega dados
        input_path = os.path.join("arquivos", "evaluated_responses.csv")
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        
        print("🔍 Colunas disponíveis:", df.columns.tolist())
        
        # Prepara e executa fine-tuning
        finetuner = ModelFinetuner()
        train_data, tokenizer = finetuner.prepare_data(df)
        finetuner.finetune(train_data, tokenizer)
        
    except Exception as e:
        print(f"❌ Erro no processo principal: {str(e)}")