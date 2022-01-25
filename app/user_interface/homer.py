import streamlit as st
from streamlit_chat import message
from app.script.chatbot_utils import BotReply


def app():
    # init messages history
    if 'message_hist' not in st.session_state:
        st.session_state.message_hist = []

    # init speaker history (who speaks at each turn ?)
    if 'speaker_hist' not in st.session_state:
        st.session_state.speaker_hist = []
        
    if 'user_message' not in st.session_state:
        st.session_state.user_message = ""


    # container for messages
    container = st.container()
    
    with container:
        message("Hey I'm Homer Bot !")
    
    def write_chat():
         with container:
                i = len(st.session_state.speaker_hist)
                for speaker, msg in zip(st.session_state.speaker_hist, st.session_state.message_hist):
                    message(msg, is_user = bool(speaker == "user"), key=i)
                    i-=1


    def actualise_chat():
        if st.session_state.user_message != "":
            # increment histories
            st.session_state.message_hist.append(st.session_state.user_message)
            st.session_state.speaker_hist.append("user")
            #add user message
            st.session_state.user_message = ""
            
            
            thread = BotReply("Homer", st.session_state.speaker_hist, st.session_state.message_hist)
            thread.start()
            write_chat()
            bot_reply = thread.join()
            
            with container:
                message(bot_reply, key=-1)
            # increment histories
            st.session_state.message_hist.append(bot_reply)
            st.session_state.speaker_hist.append("Homer")
            

    
    # 1. user writes something
    st.text_input("You: ", key="user_message", on_change=actualise_chat())


    