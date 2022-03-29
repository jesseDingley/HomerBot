import streamlit as st
from app import homer
from app.multipage import MultiPage




app = MultiPage()

# Title of the main page
st.title("HomerBot")

# Add all your applications (pages) here
app.add_page("Homer bot",homer.app)

# The main app
app.run()