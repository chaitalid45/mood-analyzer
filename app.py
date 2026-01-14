import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from recommeder import classAgent
from model_utils import predict_mood
import json

st.set_page_config(
    page_title="Mood Analyzer AI",
    page_icon="🎭",
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>🎭 AI Mood Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a face image to detect mood and get song & activity recommendations</p>", unsafe_allow_html=True)

st.divider()

img_size = 224
class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']

st.subheader("Upload Image")
uploaded_file = st.file_uploader("Choose a face image (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show uploaded image
    image = Image.open(uploaded_file)
    st.success("Image uploaded successfully!")

    # ------------------ Preprocess ------------------
    image_resized = image.resize((img_size, img_size))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ------------------ Prediction ------------------
    with st.spinner("Analyzing mood..."):
        prediction = predict_mood(img_array)
        predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"## Predicted Mood: **{predicted_class.upper()}**")

    with st.spinner("Generating recommendations..."):
        agent = classAgent(predicted_class)
        suggested_songs = agent.song_suggestion()
        suggested_activity = agent.activity_suggestion()

        songs = json.loads(suggested_songs)
        activity = json.loads(suggested_activity)

    st.divider()

    st.subheader("🎵 Song Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Marathi")
        st.write(songs["marathi_song"])

    with col2:
        st.markdown("### Hindi")
        st.write(songs["hindi_song"])

    with col3:
        st.markdown("### English")
        st.write(songs["english_song"])

    st.divider()
    st.subheader(" Activity Suggestions")

    for i, act in enumerate(activity["activities"], 1):
        st.write(f" {i}. {act}")
