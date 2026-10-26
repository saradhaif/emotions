
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Image Classification App")
st.write("Upload a picture to get a prediction.")

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    if st.button("Predict"):
        # Get the expected input shape from the model
        input_shape = input_details[0]["shape"]
        height, width = int(input_shape[1]), int(input_shape[2])

        image = image.resize((width, height))

        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Use the same preprocessing as during training
        img_array = img_array / 255.0

        # Match the model's expected input data type
        img_array = img_array.astype(input_details[0]["dtype"])

        # Run inference
        interpreter.set_tensor(
            input_details[0]["index"], img_array
        )
        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )

        st.write("Model output:", prediction)

        predicted_class = int(np.argmax(prediction, axis=1)[0])
        st.success(f"Predicted class: {predicted_class}")
