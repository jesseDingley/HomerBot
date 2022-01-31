import streamlit as st
from streamlit_chat import message
from app.chatbot.chatbot import get_response

NB_CHAT = 5

def app():
    if 'message_hist' not in st.session_state:
        st.session_state.message_hist = []

    if 'speaker_hist' not in st.session_state:
        st.session_state.speaker_hist = []
        
    if 'user_message' not in st.session_state:
        st.session_state.user_message = ""
        
    
    expander = st.expander("Message history")
    container = st.container()

    def write_history():
        if not len(st.session_state.message_hist) < NB_CHAT:
            with expander:
                message("Hey I'm Homer Bot !", key=-1)
                i = 0
                for msg, spk in zip(st.session_state.message_hist[:-NB_CHAT], st.session_state.speaker_hist[:-NB_CHAT]):
                    message(msg, bool(spk == "user"), key=i)
                    i += 1
        
    def write_chat():
        with container:
            if len(st.session_state.message_hist) < NB_CHAT:
                message("Hey I'm Homer Bot !", key=-1)
            i = -2
            for msg, spk in zip(st.session_state.message_hist[-NB_CHAT:], st.session_state.speaker_hist[-NB_CHAT:]):
                message(msg, bool(spk == "user"), key=i)
                i -= 1
    
    def actualise_chat():
        if st.session_state.user_message != "":
            st.session_state.message_hist.append(st.session_state.user_message)
            st.session_state.speaker_hist.append("user")
            st.session_state.user_message = ""
            
            bot_reply = get_response("Homer", st.session_state.message_hist, st.session_state.speaker_hist) 
            st.session_state.message_hist.append(bot_reply)
            st.session_state.speaker_hist.append("Homer")
            
            write_history()
            write_chat()
    
    
    st.text_input("You: ", key="user_message", on_change=actualise_chat())
    
        
    if 'set_chat' not in st.session_state:
        st.session_state.set_chat = True
        write_chat()
        
    if st.button(label="Reset chat", key="reset"):
        st.session_state.message_hist = []
        st.session_state.speaker_hist = []
        write_chat()
        
    page_bg_img = '''
    <style>
        .stApp {
            background-image: url("https://wallpaperaccess.com/full/1202258.jpg");
            background-size: cover;
        }
    </style>
    '''
    st.image("static/Homer_Robot_Icon.png")
    
    robot_img ='''
    <script type="text/javascript">
    let collection = document.getElementsByClassName("css-1y35n86");
    for(i=0; i < collection.length; i++){
        let nodes = collection[i].children;
        nodes[0].src = "https://www.sinoconcept.fr/wp-content/uploads/2020/07/cone-signalisation-pvc-orange-50.jpg";
    }
    </script>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    