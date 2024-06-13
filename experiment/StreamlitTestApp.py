import os
import traceback
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks import get_openai_callback

#creating title for the app
st.set_page_config(page_title="AMS Assist powered by Gen AI!!!")
st.title("AMS Assist Application")

#create a form using st.form
with st.form("user_inputs"):
	#Input Fields
	userquery=st.text_input("Ask your Questions",max_chars=500)
	
	#Add Button
	button=st.form_submit_button("Assist Me")

	# Check if the button is clicked and the input filed is not empty
	if button and userquery is not None:
		st.spinner("In progress...")