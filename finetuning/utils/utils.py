from transformers import TextIteratorStreamer
from threading import Thread
from typing import Iterator
import torch
from peft import PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)


################################################################################
# llama2-7b-chat-hf text convert function
################################################################################
def llama2_chat_text_convert_train(data_point, input_col, output_col):
    # have to use double </s> at end in order to have proper ending, not sure why yet
    instruction = f"""<s>[INST] {data_point[input_col].strip()} [/INST]""".strip()
    sentence = f"""<s>[INST] {data_point[input_col].strip()} [/INST] {data_point[output_col].strip()} </s></s>""".strip()
    response = f"""{data_point[output_col].strip()} </s></s>""".strip()
    return {"instruction": instruction, "text": sentence, "label": response}


def llama2_chat_text_convert_test(data_point, input_col):
    # have to use double </s> at end in order to have proper ending, not sure why yet
    sentence = f"""<s>[INST] {data_point[input_col].strip()} [/INST]""".strip()
    return {"text": sentence}


class llama2:
    def __init__(self, model_id):
        # self.DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        # If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
        self.DEFAULT_SYSTEM_PROMPT = """Write SQL for the question using table and schema, put the SQL in triple backticks. 
        For example: 
        ```SQL```
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.MAX_NEW_TOKENS = 1024
        self.DEFAULT_MAX_NEW_TOKENS = 1024
        self.MAX_INPUT_TOKEN_LENGTH = 4000
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
        self.history = [("system", self.DEFAULT_SYSTEM_PROMPT)]

    def get_prompt_adjust_input_length(self, message: str) -> str:
        chat_history = self.history.copy()
        system_prompt_length = len(
            self.tokenizer.encode(f'<s>[INST] <<SYS>>\n{self.DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n'))

        total_history_length = system_prompt_length + len(self.tokenizer.encode(message + " [/INST]")) + \
                               sum([len(self.tokenizer.encode(user_input + " [/INST] ")) + len(
                                   self.tokenizer.encode(response + " </s><s>[INST] "))
                                    for user_input, response in chat_history])

        while total_history_length > self.MAX_INPUT_TOKEN_LENGTH:
            user_input, response = chat_history.pop(0)
            total_history_length -= len(self.tokenizer.encode(user_input + " [/INST] ")) + len(
                self.tokenizer.encode(response + " </s><s>[INST] "))

        texts = [f'<s>[INST] <<SYS>>\n{self.DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)

    def reset_history(self):
        self.history = [("system", self.DEFAULT_SYSTEM_PROMPT)]

    def update_history(self, role: str, message: str):
        self.history.append((role, message))

    def run(self, message: str,
            temperature: float = 0.8,
            top_p: float = 0.95,
            top_k: int = 50) -> Iterator[str]:
        prompt = self.get_prompt_adjust_input_length(message)
        inputs = self.tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')

        streamer = TextIteratorStreamer(self.tokenizer,
                                        timeout=10.,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            yield ''.join(outputs).lstrip()

################################################################################
# check tunable parameters
################################################################################
def print_trainable_parameters(model, peft_config):
    """
    Prints the number of trainable parameters in the model.
    """
    # trainable_params = 0
    # all_param = 0
    # for _, param in model.named_parameters():
    #     all_param += param.numel()
    #     if param.requires_grad:
    #         trainable_params += param.numel()
    # print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


################################################################################
# progress generation function
################################################################################
def run_generator(prompt,
                  model,
                  tokenizer,
                  temperature=0.01,
                  top_p=0.95,
                  top_k=50,
                  max_new_tokens=96,
                  eos_token_id=None):
    inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')
    try:
        # falcon tokenizer has token_type_ids, llama2 tokenizer doesn't have
        del inputs["token_type_ids"]
    except KeyError:
        pass
    streamer = TextIteratorStreamer(tokenizer,
                                    timeout=10.,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs).lstrip()


def progress_generation(prompt,
                        model,
                        tokenizer,
                        temperature=0.01,
                        top_p=0.95,
                        top_k=50,
                        max_new_tokens=96,
                        eos_token_id=None,
                        show=True):
    generator = run_generator(prompt,
                              model,
                              tokenizer,
                              temperature=temperature,
                              top_p=top_p,
                              top_k=top_k,
                              max_new_tokens=max_new_tokens,
                              eos_token_id=eos_token_id)
    previous_texts = ""
    for response in generator:
        if show:
            print(response[len(previous_texts):], end='')
        previous_texts = response
    return previous_texts


################################################################################
# load fine-tuned model
################################################################################
def load_merged(model_name, quantization_config=None):
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=quantization_config,
                                                 device_map='auto',
                                                 trust_remote_code=True)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def load_base_and_adapter(base_model_id, adapter, quantization_config=None):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

