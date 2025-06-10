import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

class ModelEvaluator:
    def __init__(self, original_model_path, finetuned_model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configuração do modelo original
        self.original_tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        self.original_tokenizer.pad_token = self.original_tokenizer.eos_token
        self.original_model = AutoModelForCausalLM.from_pretrained(original_model_path).to(self.device)
        
        # Configuração do modelo fine-tuned
        self.finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path).to(self.device)

    def generate_response(self, model, tokenizer, prompt, max_length=100):
        """Gera resposta com máscara de atenção explícita"""
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Gera máscara de atenção explícita
        attention_mask = inputs.attention_mask if 'attention_mask' in inputs else None
        
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

    def compare_models(self, prompts, num_samples=1000):
        """Compara os modelos com tratamento robusto"""
        results = []
        for prompt in prompts[:num_samples]:
            try:
                original_response = self.generate_response(
                    self.original_model, 
                    self.original_tokenizer, 
                    prompt
                )
                finetuned_response = self.generate_response(
                    self.finetuned_model,
                    self.finetuned_tokenizer,
                    prompt
                )
                
                results.append({
                    "prompt": prompt,
                    "original_response": original_response,
                    "finetuned_response": finetuned_response,
                    "is_injection": "inject" in prompt.lower() or "ignore" in prompt.lower()
                })
            except Exception as e:
                print(f"⚠️ Erro no prompt '{prompt[:30]}...': {str(e)}")
                continue
                
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Configurações
    data_dir = "arquivos"
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_path = os.path.join(data_dir, "prompt_injections_dataset.csv")
    output_path = os.path.join(data_dir, "model_comparison.csv")

    try:
        # Carrega dados
        df = pd.read_csv(dataset_path, encoding='utf-8-sig')
        test_prompts = df["prompt_injections_pt"].dropna().sample(1000).tolist()
        
        # Avaliação
        evaluator = ModelEvaluator(
            original_model_path="gpt2",
            finetuned_model_path="model_finetuned"
        )
        
        results = evaluator.compare_models(test_prompts)
        results.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n🔍 Comparação Final:")
        print(results[["prompt", "original_response", "finetuned_response"]])
        print(f"\n✅ Resultados salvos em: {output_path}")

    except Exception as e:
        print(f"\n❌ Erro fatal: {str(e)}")
        if 'df' in locals():
            print("Colunas disponíveis:", df.columns.tolist())