
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Image Classification App")
st.write("Upload a picture to get a prediction.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    if st.button("Predict"):
        # Change this to the image size used during training
        image = image.resize((224, 224))

        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Apply the SAME preprocessing used during training
        img_array = img_array / 255.0

        prediction = model.predict(img_array)

        st.write("Model output:", prediction)
