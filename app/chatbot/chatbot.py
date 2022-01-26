import requests

from app.core.config import get_api_settings

settings = get_api_settings()

API_URL = settings.api_url
API_TOKEN = settings.api_token

def create_inputs(msg_history,spk_history):
    past_user_inputs = []
    generated_responses = []

    for i in range(len(msg_history)-1):
        if spk_history[i] == "user":
            past_user_inputs.append(msg_history[i])
        else:
            generated_responses.append(msg_history[i])

    text = msg_history[-1]

    payload = {"inputs": {"past_user_inputs": past_user_inputs,
                                "generated_responses": generated_responses,
                                "text": text},
               "parameters": {"top_k": 300,
                              "temperature": 0.7}
               }

    return payload

def get_response(bot,msg_history,spk_history):
    payload = create_inputs(msg_history,spk_history)
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL[bot], headers=headers, json=payload)
    bot_response = response.json()
    print()
    print()
    print(bot_response)
    return bot_response["generated_text"]
