from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

import streamlit as st 
import os
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

### Function to load Gemini pro vision
model=genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_response(input,image,prompt):
    response=model.generate_content([input,image[0],prompt])
    return response.text 

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        #Read the file into bytes
        bytes_data=uploaded_file.getvalue()
        
        image_parts=[
            
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

#initialize our streamlit app

st.set_page_config(page_title="AI FITNESS TRAINERS")

st.header("AI FITNESS TRAINERS")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file=st.file_uploader("Choose an image of the health domain....",type=["jpeg","jpg","png"])
image=""
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image.", use_column_width=True)
    
submit=st.button("Guide me about the image")

input_prompt="""
As an fitness trainers you are expert in understanding healthcare domain.
You have extreme knowledge of sciences, physiological and psychological fitness, spiritual and ayurveda.
We will upload an image of health doamin and you will have to answer any questions
based on the uploaded image in medical, scientific and evidence-based manner.
"""

# If submit button is clicked
if submit:
    image_data=input_image_details(uploaded_file)
    response=get_gemini_response(input_prompt,image_data,input)
    st.subheader("Response")
    st.write(response)