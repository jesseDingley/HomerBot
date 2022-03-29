import requests
from app.config import get_api_settings

settings = get_api_settings()

API_URL = settings.api_url
API_TOKEN = settings.api_token

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

def get_response(bot,msg_history,spk_history):
    payload = create_inputs(msg_history,spk_history)
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL[bot], headers=headers, json=payload)
    bot_response = response.json()
    print(bot_response)
    return bot_response["generated_text"]
