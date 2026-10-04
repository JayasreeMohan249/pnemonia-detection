
# ============================================================
# Pneumonia Detection - Streamlit Deployment Application
# ============================================================

# Standard libraries
import io
import os

# Image processing and numerical libraries
import cv2
import numpy as np
import pandas as pd
import pydicom

# Web application and deep learning libraries
import streamlit as st
import tensorflow as tf


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# Model Configuration
# ============================================================

# Image dimensions expected by the trained CNN
IMG_SIZE = 224

# Path to the trained model
MODEL_PATH = "custom_cnn_v2_baseline.keras"

# Class labels corresponding to model output indices
CLASS_NAMES = [
    "Normal",
    "Lung Opacity",
    "No Lung Opacity / Not Normal"
]


# ============================================================
# Load Trained CNN Model
# ============================================================

@st.cache_resource
def load_model():
    """
    Load and cache the trained CNN model.

    The function also validates the model input and output
    dimensions before allowing inference.
    """

    # Check whether the model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found."
        )

    # Load model for inference only
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # Verify that the model expects 224x224 grayscale images
    expected_input_shape = (None, IMG_SIZE, IMG_SIZE, 1)

    if model.input_shape != expected_input_shape:
        raise ValueError(
            f"Unexpected model input shape: {model.input_shape}. "
            f"Expected: {expected_input_shape}"
        )

    # Verify that the model produces three class probabilities
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model outputs {model.output_shape[-1]} classes, "
            f"but {len(CLASS_NAMES)} classes are configured."
        )

    return model


# ============================================================
# DICOM Image Preprocessing
# ============================================================

def preprocess_dicom(dicom_bytes):
    """
    Convert an uploaded DICOM chest X-ray into the format
    expected by the CNN model.
    """

    # Read DICOM image directly from uploaded bytes
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))

    # Extract image pixel values and convert to float32
    image = ds.pixel_array.astype(np.float32)

    # MONOCHROME1 stores brighter pixels using lower values.
    # Invert such images so they follow the expected X-ray appearance.
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    # --------------------------------------------------------
    # Min-Max Normalization
    # --------------------------------------------------------

    # Shift minimum pixel intensity to zero
    image = image - image.min()

    max_value = image.max()

    # Scale pixel values to the range [0, 1]
    if max_value > 0:
        image = image / max_value

    # Create an 8-bit version only for display in Streamlit
    display_image = (
        image * 255
    ).clip(0, 255).astype(np.uint8)

    # Resize image to the dimensions used during model training
    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Add grayscale channel:
    # (224, 224) -> (224, 224, 1)
    image = np.expand_dims(image, axis=-1)

    return image.astype(np.float32), display_image


# ============================================================
# Initialize Model
# ============================================================

try:
    # Load the model once and cache it for future predictions
    model = load_model()

except Exception as e:
    # Stop the application if the trained model cannot be loaded
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# Streamlit User Interface
# ============================================================

st.title("🩺 Pneumonia Detection from Chest X-ray")

st.write(
    "Upload a chest X-ray in DICOM (`.dcm`) format. "
    "The trained CNN will classify the image into one of "
    "three categories."
)

# Display the classes supported by the model
st.caption(
    "Classes: Normal • Lung Opacity • "
    "No Lung Opacity / Not Normal"
)

# Allow users to upload only DICOM files
uploaded_file = st.file_uploader(
    "Upload DICOM Chest X-ray",
    type=["dcm"]
)


# ============================================================
# Process Uploaded Image
# ============================================================

if uploaded_file is not None:

    try:

        # Read uploaded DICOM file as bytes
        dicom_bytes = uploaded_file.getvalue()

        # Preprocess the DICOM image for model inference
        processed_image, display_image = preprocess_dicom(
            dicom_bytes
        )

        # Display the uploaded chest X-ray
        st.image(
            display_image,
            caption="Uploaded Chest X-ray",
            clamp=True
        )

        # Add batch dimension required by TensorFlow:
        # (224, 224, 1) -> (1, 224, 224, 1)
        input_image = np.expand_dims(
            processed_image,
            axis=0
        )

        # Validate the final input shape before prediction
        expected_shape = (
            1,
            IMG_SIZE,
            IMG_SIZE,
            1
        )

        if input_image.shape != expected_shape:
            raise ValueError(
                f"Unexpected image shape: {input_image.shape}. "
                f"Expected: {expected_shape}"
            )


        # ====================================================
        # CNN Prediction
        # ====================================================

        with st.spinner("Analyzing chest X-ray..."):

            # Generate class probabilities from the CNN
            predictions = model.predict(
                input_image,
                verbose=0
            )

        # Extract probabilities for the uploaded image
        probabilities = predictions[0]

        # Identify the class with the highest probability
        predicted_class_idx = int(
            np.argmax(probabilities)
        )

        predicted_class_name = (
            CLASS_NAMES[predicted_class_idx]
        )

        # Confidence score of the predicted class
        confidence = float(
            probabilities[predicted_class_idx]
        )


        # ====================================================
        # Prediction Result
        # ====================================================

        st.divider()
        st.subheader("Prediction Result")


        # ----------------------------------------------------
        # Class 1: Normal
        # ----------------------------------------------------

        if predicted_class_name == "Normal":

            st.success("✅ No Pneumonia Detected")

            st.write(
                "The chest X-ray has been classified as "
                "**Normal**. The model did not identify "
                "lung opacity associated with pneumonia."
            )


        # ----------------------------------------------------
        # Class 2: Lung Opacity
        # ----------------------------------------------------

        elif predicted_class_name == "Lung Opacity":

            st.error("⚠️ Possible Pneumonia Detected")

            st.write(
                "The chest X-ray has been classified as "
                "**Lung Opacity**. This finding may be "
                "associated with pneumonia."
            )

            st.warning(
                "Further medical evaluation is recommended. "
                "This AI prediction should not be considered "
                "a confirmed medical diagnosis."
            )


        # ----------------------------------------------------
        # Class 3: No Lung Opacity / Not Normal
        # ----------------------------------------------------

        elif predicted_class_name == "No Lung Opacity / Not Normal":

            st.warning("⚠️ Abnormal Chest X-ray Detected")

            st.write(
                "The model did **not identify lung opacity "
                "associated with pneumonia**, but the chest "
                "X-ray was classified as **Not Normal**."
            )

            st.info(
                "The image may contain other abnormalities "
                "that this model is not designed to identify. "
                "Please consult a healthcare professional "
                "for further evaluation."
            )


        # ====================================================
        # Prediction Confidence
        # ====================================================

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2%}"
        )


        # ====================================================
        # Class Probability Distribution
        # ====================================================

        st.subheader("Class Probabilities")

        # Create a table containing probability for each class
        probability_df = pd.DataFrame(
            {
                "Class": CLASS_NAMES,
                "Probability": probabilities
            }
        ).sort_values(
            by="Probability",
            ascending=False
        )

        # Display probabilities as percentages
        st.dataframe(
            probability_df.style.format(
                {"Probability": "{:.2%}"}
            ),
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # Model Information
        # ====================================================

        with st.expander("Model Information"):

            st.write("**Model:** Custom CNN")
            st.write(f"**Input shape:** {model.input_shape}")
            st.write(f"**Output shape:** {model.output_shape}")

            st.write(
                "**Image preprocessing:** "
                "Grayscale → Min-max normalization → "
                "224 × 224 resize → single channel"
            )


    # Handle invalid or unsupported DICOM files
    except Exception as e:

        st.error(
            "Unable to process this DICOM image. "
            "Please make sure you uploaded a valid "
            "chest X-ray DICOM file."
        )

        st.exception(e)


# ============================================================
# Medical Disclaimer
# ============================================================

st.divider()

st.caption(
    "⚠️ This application is an academic Computer Vision project "
    "and is not intended to replace professional medical diagnosis."
)
