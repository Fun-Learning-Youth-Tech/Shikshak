from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from typing import Dict, List
import logging

class TeacherBot:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Add special tokens for different languages and educational contexts
        special_tokens = {
            "additional_special_tokens": [
                "[EN]", "[ES]", "[FR]", "[DE]", "[ZH]",  # Language tokens
                "[EXPLAIN]", "[QUIZ]", "[FEEDBACK]"       # Education-specific tokens
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logging.basicConfig(
            filename='logs/model.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def preprocess_data(self, dataset: Dict) -> Dict:
        """Preprocess the dataset for training"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        return tokenized_dataset

    def train(self, dataset: Dict, output_dir: str = "./trained_model"):
        """Train the model on the provided dataset"""
        try:
            tokenized_dataset = self.preprocess_data(dataset)
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=100,
                evaluation_strategy="steps",
                save_strategy="steps",
                save_steps=1000,
                eval_steps=1000
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"]
            )

            trainer.train()
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logging.info(f"Model training completed and saved to {output_dir}")
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def generate_response(self, prompt: str, language: str = "EN") -> str:
        """Generate a response for the given prompt in the specified language"""
        try:
            # Add language token to prompt
            lang_prompt = f"[{language}] {prompt}"
            
            inputs = self.tokenizer(lang_prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"Generated response for prompt: {prompt[:50]}...")
            return response
            
        except Exception as e:
            logging.error(f"Generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"