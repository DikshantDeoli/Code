import os
import json
import time
import logging
import pandas as pd
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
import bitsandbytes as bnb
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
from peft import PeftModel



# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join("./logs", "training_Mistral_Instruct_2k.log")),
        logging.StreamHandler()
    ]
)

class CustomError(Exception):
    pass

def fine_tune_mistral(dataset_file, output_dir):
    try:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )


        model_id = "HuggingFaceH4/zephyr-7b-beta"

        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
        tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

        df_test = pd.read_csv(dataset_file)
        dataset = Dataset.from_pandas(df_test)
        
       

        def generate_prompt(data_point):
                
                prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                        'appropriately completes the request.\n\n'
                if data_point['Question']:
                    text = f"""<s>[INST]{prefix_text} {data_point["Question"]} here are the inputs {data_point["Context"]} [/INST]{data_point["Open_AI"]}</s>"""
                else:
                    text = f"""<s>[INST]{prefix_text} {data_point["Question"]} [/INST]{data_point["Open_AI"]} </s>"""
                
                return text

        
        text_column = [generate_prompt(data_point) for data_point in dataset]

        dataset = dataset.add_column("prompt", text_column)

        # dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
        dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
        dataset = dataset.train_test_split(test_size=0.1)
        train_data = dataset["train"]
        test_data = dataset["test"]


        print("*"*25)
        print(train_data)
        print("*"*25)
        print(test_data)
        print("*"*25)

        print("*"*25)
        print(dataset)
        print("*"*25)

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        
        def find_all_linear_names(model):
            cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])
                if 'lm_head' in lora_module_names: # needed for 16-bit
                    lora_module_names.remove('lm_head')
            return list(lora_module_names)
        
        modules = find_all_linear_names(model)

        print(modules)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # model = get_peft_model(model, lora_config)

        # trainable, total = model.get_nb_trainable_parameters()
        
        # print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

        tokenizer.pad_token = tokenizer.eos_token
        torch.cuda.empty_cache()

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=test_data,
            dataset_text_field="prompt",
            peft_config=lora_config,
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                optim="paged_adamw_8bit",
                save_strategy="epoch",
                logging_steps=1,
                max_steps=400,
                learning_rate=2e-4,
                # bf16=True,
                # tf32=True,
                # max_grad_norm=0.3,
                warmup_steps=0.03,
                # gradient_checkpointing=True,
                # group_by_length=False,
                # lr_scheduler_type="constant",
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        
        
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()

        # new_model = "mistralai-Code-Instruct-Finetune-test" 

        trainer.model.save_pretrained(output_dir)




    except CustomError as e:
        logging.error(f"Custom error: {str(e)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

# Example usage:
dataset_file = "./data/new_dataset_2k_aprox_openAI.csv"
output_directory = "./model/Zyper_2k/"
fine_tune_mistral(dataset_file, output_directory)