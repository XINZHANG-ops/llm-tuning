from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model
from trl import SFTTrainer
################################################################################
# TrainingArguments parameters
################################################################################
# Number of training epochs
num_train_epochs = 1

# Number of training steps (overrides num_train_epochs)
max_steps = 1500

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 2

# Batch size per GPU for evaluation
per_device_eval_batch_size = 2

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "cosine"

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.05

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 1000

# Log every X updates steps
logging_steps = 1

# max_length for truncation for oom error
max_length = 512


def customizer_train_args(output_dir):
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )
    return training_arguments


def transformer_trainer(model, peft_config, tokenizer, train_dataset, dataset_text_field="text", label_text_field=False, output_dir="./results"):
    training_arguments = customizer_train_args(output_dir)
    model = get_peft_model(model, peft_config)

    if not label_text_field:
        def tokenize_data(data_point):
            tokenized_full_prompt = tokenizer(data_point[dataset_text_field], padding=True, truncation=True, max_length=max_length)
            return tokenized_full_prompt

        train_dataset = train_dataset.map(tokenize_data)
    else:
        def tokenize_data(data_point):
            instruction = tokenizer(data_point["instruction"], padding=True, truncation=True, max_length=max_length)
            full_prompt = tokenizer(data_point[dataset_text_field], padding=True, truncation=True, max_length=max_length)
            input_ids = full_prompt["input_ids"]
            attention_mask = full_prompt["attention_mask"]
            response = tokenizer(data_point["label"], padding=True, truncation=True, max_length=max_length)
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"][1:] # this is wired no idea why llama2 tokenizer always has id 1 at begining
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        train_dataset = train_dataset.map(tokenize_data, remove_columns=train_dataset.column_names)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
        # data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    return trainer


def sft_trainer(model, peft_config, tokenizer, train_dataset, dataset_text_field="text", output_dir="./results"):
    training_arguments = customizer_train_args(output_dir)

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field=dataset_text_field,
        max_seq_length=max_length, #None, if too large will cause cuda oom
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False
    )
    return trainer
