# Engorgio (Dong et al., ICLR 2025, "An Engorgio Prompt Makes Large Language Model Babble on").
# Adapted by the ContamBench authors from https://github.com/jianshuod/Engorgio-prompt
# (commit 5fb50dd). The upstream repository carries no license file; see THIRD_PARTY_NOTICES.md.
import numpy as np
from contambench.gen.models import Model

import torch


class TemplateFactory():
    '''
        1. Use a sentence to get template and then encode
        2. Extract the template part of the encoded
    '''
    def __init__(self, model: Model,  model_name, trigger_token_length, tokenizer, embedding) -> None:
        self.model_name = model_name
        self.trigger_token_length = trigger_token_length
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.model = model
        self.add_additional_prompt("")
    
    def add_additional_prompt(self, prefix_sentence):
        # conv : templates.Conversation = templates.conv_templates[convert_name.get(self.model_name, "naive")].copy()
        
        if prefix_sentence != "" or 'alpaca' in self.model_name: 
            prefix_sentence += ' '
        ## this is where seed prompt can be added but finding appropriate prefix length later is challenging
        demo_sentence = self.tokenizer.decode([7993] * self.trigger_token_length) 
        # conv.append_message(conv.roles[0], prefix_sentence + demo_sentence)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        # print("promt from conv. =", prompt)
        prompt = self.model.add_chat_template(prefix_sentence + demo_sentence)
        # print("prompt2 = ", prompt2)
        # assert(prompt == prompt2)
        self.set_template(prompt)


    def set_template(self, prompt):
        tokenizer = self.tokenizer
        trigger_token_length = self.trigger_token_length
        embedding = self.embedding
        
        # For meta-llama and models with chat template, the prompt string already contains BOS
        # So we should NOT add it again during encoding. Use add_special_tokens=False
        if 'meta-llama' in self.model_name.lower() or '<|begin_of_text|>' in prompt:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        else:
            input_ids = tokenizer.encode(prompt)
        
        # print(input_ids)
        total_length = len(input_ids)

        if 7993 not in input_ids:
            raise ValueError(
                f"Engorgio's template placeholder (token id 7993, repeated) does not survive a decode/encode "
                f"round trip with the {self.model_name} tokenizer, so the trigger position cannot be located. "
                "The method was used with Llama-3-family tokenizers; this model is not supported.")
        prefix_len = max(index for index, item in enumerate(input_ids) if item == 7993) - trigger_token_length + 1
        self.prefix_tokens = input_ids[:prefix_len]
        self.tail_tokens = input_ids[prefix_len+trigger_token_length:]
        print("before: ", len(input_ids))
        
        input_ids[prefix_len:prefix_len + trigger_token_length] = np.random.randint(0, self.model.total_vocab_size, size = trigger_token_length).tolist()
        print("after", len(input_ids))
        print(self.model.decode_from_ids(input_ids))

        self.prefix_embeds = embedding[input_ids[:prefix_len]].detach().unsqueeze(0)
        self.tail_embeds = embedding[input_ids[prefix_len+trigger_token_length:]].detach().unsqueeze(0)
        
        self.template_length = total_length - trigger_token_length
        self.response_offset = prefix_len+trigger_token_length
        self.prefix_length = prefix_len
        self.template_w_trigger_length = total_length

    def get_input_embeddings(self, inputs_embeds):
        front_part = inputs_embeds[:, :self.trigger_token_length]
        tail_part = inputs_embeds[:, self.trigger_token_length:]
        concated = torch.concat(
            [self.prefix_embeds, front_part, self.tail_embeds, tail_part], dim=1)
        return concated
    
    def get_input_tokens(self, inputs_tokens):
        # print(self.tokenizer.decode(self.prefix_tokens + inputs_tokens + self.tail_tokens))
        return self.prefix_tokens + inputs_tokens + self.tail_tokens
    

