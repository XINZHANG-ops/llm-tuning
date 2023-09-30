from transformers import AutoTokenizer
from peft import (
    TaskType,
    LoraConfig,
    PromptTuningConfig, PromptTuningInit,
    PromptEncoderConfig, PromptEncoderReparameterizationType,
    PrefixTuningConfig
)


################################################################################
# tokenizer
################################################################################
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

################################################################################
# data parameters
################################################################################
random_seed = 42

################################################################################
# prompt tuning parameters
################################################################################
prompt_tuning_num_virtual_tokens = 100
hard_prompt_content = "Please convert the question to SQL"

# Soft Prompt
soft_prompt_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=prompt_tuning_num_virtual_tokens)

# Hard Prompt
hard_prompt_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                        prompt_tuning_init=PromptTuningInit.TEXT,
                                        prompt_tuning_init_text=hard_prompt_content,
                                        num_virtual_tokens=len(tokenizer(hard_prompt_content)["input_ids"]),
                                        tokenizer_name_or_path=model_name)


################################################################################
# p-tuning parameters
################################################################################
p_tuning_num_virtual_tokens = 100

encoder_reparameterization_type = PromptEncoderReparameterizationType.LSTM # PromptEncoderReparameterizationType.LSTM MLP

p_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=p_tuning_num_virtual_tokens,
                               encoder_reparameterization_type=encoder_reparameterization_type,
                               encoder_dropout=0.1,
                               encoder_num_layers=5,
                               encoder_hidden_size=2048)

################################################################################
# prefix-tuning parameters
################################################################################
prefix_tuning_num_virtual_tokens = 100
prefix_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=prefix_tuning_num_virtual_tokens, prefix_projection=True)


################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 16

# Alpha parameter for LoRA scaling
lora_alpha = 32

# falcon: ["query_key_value"]
# llama2: ["q_proj", "v_proj", "k_proj", "o_proj"]
target_modules = ["q_proj", "v_proj", "k_proj"]# ["query_key_value"] # ["q_proj", "v_proj", "k_proj"] # this might be adjusted for different base model, print model for layer names

modules_to_save = ["embed_tokens", "lm_head"]

# Dropout probability for LoRA layers
lora_dropout = 0.1

# Load LoRA configuration
# https://www.reddit.com/r/LocalLLaMA/comments/15fhf33/why_does_the_model_refuse_to_predict_eos/?rdt=40094
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules,
    modules_to_save=modules_to_save,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)