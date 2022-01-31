import requests
import threading
from app.core.config import get_api_settings

settings = get_api_settings()

API_URL = settings.api_url
API_TOKEN = settings.api_token

def create_inputs(msg_history,spk_history):
    past_user_inputs = []
    generated_responses = []
    context = 5
    temperature = 0.7
    top_k = 300
    
    
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
                              "temperature": temperature}
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


class BotReply (threading.Thread):
        def __init__(self, name, speaker_history, message_history):
            threading.Thread.__init__(self)
            self.name = name
            self.message_history = message_history
            self.speaker_history = speaker_history
            self._return = None
            
        def run(self):
            self._return = get_response(self.name, self.message_history, self.speaker_history)
        
        def join(self, *args):
            threading.Thread.join(self, *args)
            return self._return
