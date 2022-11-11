import streamlit as st
from streamer import streamer_webcam_gg

# set
st.title("Yes, Laughing")
# st.set_page_config(layout= 'wide')

# start
streamer_webcam_gg()