
# 🩺 Pneumonia Detection from Chest X-rays

A Computer Vision application that uses a custom Convolutional Neural Network (CNN) to classify chest X-ray images in DICOM (`.dcm`) format.

The application is built using **TensorFlow** and **Streamlit** and can be run using **GitHub Codespaces**.

## Classification Classes

The model classifies chest X-rays into three categories:

1. **Normal**
2. **Lung Opacity** — may indicate pneumonia
3. **No Lung Opacity / Not Normal** — abnormal chest X-ray without pneumonia-associated lung opacity

## Model Details

- **Model:** Custom CNN
- **Input:** `224 × 224 × 1`
- **Image type:** Grayscale chest X-ray
- **Output classes:** 3
- **Framework:** TensorFlow / Keras

## Run Using GitHub Codespaces

### 1. Open Codespaces

From the GitHub repository:

**Code → Codespaces → Create codespace on main**

### 2. Create Python 3.12 Environment

```bash
sudo apt update
sudo apt install -y python3.12-venv

rm -rf .venv
/usr/bin/python3.12 -m venv .venv
source .venv/bin/activate
