from dotenv import load_dotenv
load_dotenv() ## loading all the environments variables

import streamlit as st 
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## FUNCTION TO LOAD GEMINI MODEL AND GET RESPONSE
model=genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(question):
    response=model.generate_content(question)
    return response.text

## inintialize our streamlit app

st.set_page_config(page_title="Q&A Chatbot")

st.header("Gemini LLM Application")

input=st.text_input("Input:",key="input")
submit=st.button("Ask the Question")

## When submit is clicked

if submit:
    response=get_gemini_response(input)
    st.subheader("Response")
    st.write(response)