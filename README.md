# 🎭 AI Mood Analyzer (Image-Based Emotion Recognition)

A Deep Learning-based project that analyzes a user's mood from facial 
images and provides personalized **song recommendations** and 
**activity suggestions** using a **Streamlit web application**.

This project combines **Computer Vision ** + **Deep Learning** + **GenerativeAI** + 
**Web App deployment** into a complete end-to-end AI system.

------------------------------------------------------------------------

## 🚀 Features

-   📷 Upload a face image and detect emotion\
-   🧠 Emotion recognition using **MobileNetV2 (Transfer Learning)**\
-   🎵 Recommends real film songs (Marathi, Hindi, English)\
-   🧘 Suggests mood-based activities\
-   🌐 Interactive web interface using **Streamlit**\
-   🔐 Secure API handling using environment variables

------------------------------------------------------------------------

## 🧠 Moods Detected

The model classifies facial expressions into: 
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😄 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   🐍 Python\
-   TensorFlow / Keras\
-   MobileNetV2 (Transfer Learning)\
-   NumPy\
-   Streamlit\
-   OpenAI API (for recommendations)

------------------------------------------------------------------------

## 📂 Project Structure

    mood-analyzer-project/
    │
    ├── app.py                 # Streamlit app
    ├── train_model.py         # Model training code
    ├── model_utils.py         # Prediction helper functions
    ├── recommender.py         # Song & activity recommendation logic
    ├── requirements.txt
    ├── README.md
    │
    ├── model/
    │   └── mood_model.h5      # Trained model

------------------------------------------------------------------------

## ⚙️ How It Works

1.  User uploads a facial image \
2.  Image is preprocessed and passed to the trained CNN model\
3.  Model predicts the emotion \
4.  Based on the predicted mood:
    -   🎵 Songs are recommended\
    -   🧘 Activities are suggested\
5.  Results are displayed in a Streamlit web interface.

------------------------------------------------------------------------


## ▶️ How to Run the Project

### 1. Install dependencies

``` bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

``` bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually
http://localhost:8501).

------------------------------------------------------------------------

## 📸 Sample Output

-   Upload a face image\
-   Get mood prediction (e.g., Happy)\
-   Receive:
    -   🎵 Marathi, Hindi, English song suggestions\
    -   🧘 3 mood-based activities

------------------------------------------------------------------------

## 🔐 API Key Security

API keys are stored securely using environment variables and are **not
exposed in the code or GitHub repository**.

------------------------------------------------------------------------

## 📈 Future Improvements

-   Improve accuracy using larger datasets\
-   Use Vision Transformers (ViT)\
-   Spotify API integration for direct music playback\
-   Live webcam emotion detection\
-   Cloud deployment (Streamlit Cloud / Hugging Face Spaces)

------------------------------------------------------------------------

## 👩‍💻 Author

**Chaitali Deshpande**\
Aspiring Machine Learning / AI Engineer

------------------------------------------------------------------------

## ⭐ Why this project is valuable

This project demonstrates: 
- Real-world use of Deep Learning\
- Transfer Learning best practices\
- End-to-end ML pipeline\
- Clean project structure\
- Practical application with UI\
- Industry-relevant tools
