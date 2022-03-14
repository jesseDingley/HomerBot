import streamlit as st
from app.script import homer
from app.script.multipage import MultiPage
import streamlit.components.v1 as components




app = MultiPage()

# Title of the main page
st.title("HomerBot")

# Add all your applications (pages) here
app.add_page("Homer bot",homer.app)

# The main app
app.run()

robot_img ="""
    <script type=javascript>
        let collection = document.getElementsByTagName("iframe");
        for(i=0; i < collection.length; i++){
            let images = collection[i].contentDocument.images;
            if (images[0].src == "https://avatars.dicebear.com/api/bottts/42.svg"){
                images[0].src = "https://www.sinoconcept.fr/wp-content/uploads/2020/07/cone-signalisation-pvc-orange-50.jpg";    
            }
        console.log("Streamlit runs JavaScript"); 
        }
    </script>
    """
components.html(robot_img)