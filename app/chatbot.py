import requests
from app.config import get_api_settings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

settings = get_api_settings()

API_URL = settings.api_url
API_TOKEN = settings.api_token
API_MODEL = settings.api_model

tokenizer = AutoTokenizer.from_pretrained(API_MODEL)
model = AutoModelForCausalLM.from_pretrained(API_MODEL)

def create_inputs(msg_history,spk_history):
    past_user_inputs = []
    generated_responses = []
    context = 5
    temperature = 0.8
    top_k = 100
    top_p = 0.7
    max_length = 500
    no_repeat_ngram_size = 3
    do_sample = True
    
    
    
    for i in range(len(msg_history)-1):
        if spk_history[i] == "user":
            past_user_inputs.append(msg_history[i])
        else:
            generated_responses.append(msg_history[i])
    
    if len(past_user_inputs) > context:
        past_user_inputs = past_user_inputs[-context:]
        generated_responses = generated_responses[-context:]
    
    text = msg_history[-1]

    payload = {"inputs": {"past_user_inputs": past_user_inputs,
                                "generated_responses": generated_responses,
                                "text": text},
               "parameters": {"top_k": top_k,
                              "temperature": temperature,
                              "top_p": top_p,
                              "max_length": max_length,
                              "no_repeat_ngram_size": no_repeat_ngram_size,
                              "do_sample": do_sample}
               }

    return payload

# def get_response(bot,msg_history,spk_history):
#     payload = create_inputs(msg_history,spk_history)
#     headers = {"Authorization": f"Bearer {API_TOKEN}"}
#     response = requests.post(API_URL[bot], headers=headers, json=payload)
#     bot_response = response.json()
#     print(bot_response)
#     return bot_response["generated_text"]


def get_response(bot,msg_history,spk_history):
    past_user_inputs = []
    generated_responses = []
    context = 5
    
    for i in range(len(msg_history)-1):
        if spk_history[i] == "user":
            past_user_inputs.append(msg_history[i])
        else:
            generated_responses.append(msg_history[i])
    
    if len(past_user_inputs) > context:
        past_user_inputs = past_user_inputs[-context:]
        generated_responses = generated_responses[-context:]
    
    text = msg_history[-1]
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    #new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    #bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if len(past_user_inputs) == 1 else new_user_input_ids
    
    bot_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    for i in range(len(generated_responses)-1,-1,-1):
        new_user_input = tokenizer.encode(past_user_inputs[i] + tokenizer.eos_token, return_tensors='pt')
        new_bot_input = tokenizer.encode(generated_responses[i] + tokenizer.eos_token, return_tensors='pt')
        
        bot_input_ids = torch.cat([bot_input_ids, new_user_input,new_bot_input], dim=-1)

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=1000,                    
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=3,
        # # num_beams = 50       
        do_sample=True, 
        top_k=100, 
        top_p=0.7,
        # length_penalty = 0.5,
        temperature = 0.8
        # repetition_penalty=1.3 
    )
    
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

