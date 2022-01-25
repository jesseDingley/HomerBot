import threading
from app.chatbot.chatbot import get_response


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