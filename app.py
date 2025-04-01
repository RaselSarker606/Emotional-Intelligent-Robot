import streamlit as st
import cv2
import google.generativeai as genai
import pyttsx3
import numpy as np
from PIL import Image
import os


#api configuration (gemini api) ======================
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAqtUlUwuP_fe0nfT1he3qmKb6yJjhlLds'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('models/gemini-1.5-flash-latest')


# custom function ===========================
def generate_image_response(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    prompt = "Analyze the emotions conveyed by the person in this image and respond directly to them, using 'you' in a short and empathetic way."
    response = model.generate_content([prompt, img])

    return response.text

def speak_response(response_text):
    # initialize text to speech engine
    tts_engine = pyttsx3.init()
    tts_engine.say(response_text)
    tts_engine.runAndWait()


# app here =================================
st.title("Emotional Intelligent-Robot")
st.subheader("It will detect your emotional from live video")


# working with camera =============================
camera = cv2.VideoCapture(0)
while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
        st.write("Unable to capture video")
        break

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels='RGB')

    response = generate_image_response(frame)

    if response:
        st.write(response)
        speak_response(response)

    cv2.waitKey(1)

camera.release()
