import os
import pandas as pd
from transformers import pipeline, AutoTokenizer
import torch
from tqdm import tqdm

class ResponseGenerator:
    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(), "arquivos")
        self.input_path = os.path.join(self.data_dir, "prompt_injections_dataset.csv")
        self.output_path = os.path.join(self.data_dir, "generated_responses.csv")
        
    def initialize_generator(self):
        """Configura o gerador de respostas"""
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            return pipeline(
                "text-generation",
                model="gpt2",
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except Exception as e:
            print(f"❌ Falha ao inicializar modelo: {str(e)}")
            raise

    def safe_generate(self, generator, prompt):
        """Geração segura de respostas"""
        try:
            return generator(
                prompt,
                max_length=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )[0]["generated_text"].replace(prompt, "").strip()
        except:
            return "[ERRO NA GERAÇÃO]"

    def process_dataset(self):
        """Processa o dataset completo"""
        try:
            # Carrega dados
            df = pd.read_csv(self.input_path, encoding='utf-8-sig')
            print(f"📊 Total de prompts: {len(df)}")
            
            # Inicializa modelo
            generator = self.initialize_generator()
            
            # Gera respostas
            results = []
            for _, row in tqdm(df.iterrows(), total=len(df)):
                prompt = str(row["prompt_injections_pt"])
                response = self.safe_generate(generator, prompt)
                
                results.append({
                    "prompt_original": row["prompt_injections"],
                    "prompt_traduzido": prompt,
                    "resposta": response,
                    "tipo": "injeção" if "injeção" in prompt.lower() else "normal"
                })
            
            # Salva resultados
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            print(f"\n💾 Resultados salvos em: {self.output_path}")
            return results_df
            
        except Exception as e:
            print(f"❌ Erro no processamento: {str(e)}")
            raise

if __name__ == "__main__":
    rg = ResponseGenerator()
    results = rg.process_dataset()
    print("\nExemplo de resultado:")
    print(results.iloc[0]["resposta"])