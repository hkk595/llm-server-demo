import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline


def llm_fine_tune(pretrained_model: str, train_dataset: str, output_path: str, app):
    # Load the model
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model,
                                                     quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                                                            bnb_4bit_compute_dtype=getattr(
                                                                                                torch, "float16"),
                                                                                            bnb_4bit_quant_type="nf4"))
    llm_model.config.use_cache = False
    llm_model.config.pretraining_tp = 1

    # Load the tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model,
                                                  trust_remote_code=True)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "right"

    # Set the training arguments
    training_arguments = TrainingArguments(output_dir=output_path, per_device_train_batch_size=4, max_steps=100)

    # Create Supervised Fine-Tuning trainer
    llm_sft_trainer = SFTTrainer(model=llm_model,
                                 args=training_arguments,
                                 train_dataset=load_dataset(path=train_dataset, split="train"),
                                 tokenizer=llm_tokenizer,
                                 peft_config=LoraConfig(task_type="CAUSAL_LM", r=64, lora_alpha=16, lora_dropout=0.1),
                                 dataset_text_field="text")

    # Train and save the model
    llm_sft_trainer.train()
    llm_sft_trainer.save_model(output_path)
    app.model = llm_model
    app.tokenizer = llm_tokenizer

    app.model_trained = True


def llm_prompt(user_prompt: str, llm_model, llm_tokenizer):
    text_generation_pipeline = pipeline(task="text-generation", model=llm_model, tokenizer=llm_tokenizer,
                                        max_length=300)
    model_answer = text_generation_pipeline(user_prompt)
    return model_answer[0]['generated_text']
