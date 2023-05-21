import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

# Load the pre-trained model
model = keras.models.load_model("app/optimal_model.h5")
# Load the label encoder
label_encoder = pickle.load(open("app/label_encoder.pkl", "rb"))
num_classes = len(label_encoder.classes_)  # Change this to the number of output classes in your model

def dog_class_prediction(image):
    """
    A function that takes an input image of a dog and returns the predicted breed with a probability score.

    Parameters:
    image (numpy.ndarray): A numpy array representing the input image of a dog.

    Returns:
    dict: A dictionary containing the predicted dog breed as keys and the corresponding probability score as values.
    """

    # Convert input image to RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    dim = (250, 250)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    # Perform histogram equalization on the image
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_equ = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # Apply non-local means filter on test image
    dst_img = cv2.fastNlMeansDenoisingColored(
        src=img_equ,
        dst=None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21)

    # Convert the modified image to a numpy array
    img_array = keras.preprocessing.image.img_to_array(dst_img)

    # Apply preprocess Xception
    img_array = img_array.reshape((-1, 250, 250, 3))
    img_array = tf.keras.applications.xception.preprocess_input(img_array)

    # Make a prediction
    prediction = model.predict(img_array).flatten()

    # Create a dictionary with the predicted dog breed and the corresponding probability score
    result = {label_encoder.inverse_transform([i])[0]: float(prediction[i]) for i in range(num_classes)}

    return result


def main():
    st.title("Dog Breed Classification")

    # Sidebar
    st.sidebar.title("Dog Breed Classifier")
    st.sidebar.write("Upload an image of a dog to classify its breed.")
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(image, channels="BGR", use_column_width=True)

        # Make a prediction
        prediction = dog_class_prediction(image)

        # Display the most likely breed
        sorted_predictions = sorted(prediction.items(), key=lambda x: x[1], reverse=True)
        most_likely_breed = sorted_predictions[0][0]
        st.write(f"**<span style='font-size: 40px; color: #045FB4;'>This dog is likely a {most_likely_breed}</span>**", unsafe_allow_html=True)

        # Display the top four predicted dog breeds and probability scores
        st.subheader("Top Predictions:")
        sorted_predictions = sorted_predictions[:4]
        for i, (breed, score) in enumerate(sorted_predictions):
            if i == 0:
                st.write(f"**{i+1}. {breed}: {score:.4f}**")
            else:
                st.write(f"{i+1}. {breed}: {score:.4f}")



if __name__ == "__main__":
    main()
