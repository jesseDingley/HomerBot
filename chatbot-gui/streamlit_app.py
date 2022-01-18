import streamlit as st
from streamlit_chat import message



# define bot
def bot(history):
    return " ".join(history).upper()




# init messages history
if 'message_hist' not in st.session_state:
    st.session_state.message_hist = []

# init speaker history (who speaks at each turn ?)
if 'speaker_hist' not in st.session_state:
    st.session_state.speaker_hist = []




# place holder for chat bot div
placeholder = st.empty()



# 1. user writes something
input_ = st.text_input("You: ")
# increment histories
st.session_state.message_hist.append(input_)
st.session_state.speaker_hist.append("user")


# 2. then the bot replies
bot_reply = bot(st.session_state.message_hist)
# increment histories
st.session_state.message_hist.append(bot_reply)
st.session_state.speaker_hist.append("bot")





# show chat history
with placeholder.container():
    message("Hey there bro!")
    for speaker, msg in zip(st.session_state.speaker_hist, st.session_state.message_hist):
        if msg != "":
            if speaker == "user":
                message(msg, is_user = True)
            else:
                message(msg)