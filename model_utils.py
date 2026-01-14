import tensorflow as tf
import numpy as np

IMG_SIZE = 224
class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']

model = tf.keras.models.load_model("model/mood_model.h5")

def predict_mood(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]

