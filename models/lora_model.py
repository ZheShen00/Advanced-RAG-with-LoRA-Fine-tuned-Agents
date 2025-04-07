from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

class LoRAModel:
    def __init__(self, base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", 
                 lora_weights_path="lora_mc_model"):
        """
        Initializing the LoRA trim model
        
        Parameters:
            base_model_name: base model name
            lora_weights_path: LoRA weights path
        """
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Inspection of equipment and loading of models 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for LoRA model")
        
        # Loading the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Loading LoRA weights
        self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
        
        print(f"LoRA model loaded successfully: {base_model_name} with weights from {lora_weights_path}")
    
    def generate(self, prompt, max_new_tokens=100, temperature=0.3):
        """Generating text using the LoRA model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_beams=3,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt
        if prompt in response:
            response = response.replace(prompt, "").strip()
            
        return response