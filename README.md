---
title: Pneumonia Detection
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
---

# Pneumonia Detection from Chest X-rays

This application uses a custom Convolutional Neural Network (CNN)
to classify chest X-ray DICOM images.

## Classes

1. Normal
2. Lung Opacity
3. No Lung Opacity / Not Normal

## Model Input

The model expects grayscale chest X-ray images with shape:

`224 × 224 × 1`

## Preprocessing

The uploaded DICOM image is:

1. Read using pydicom
2. Converted to float32
3. Min-max normalized
4. Resized to 224 × 224
5. Passed to the CNN as a single-channel grayscale image

## Disclaimer

This application is an academic Computer Vision project and is
not intended to replace professional medical diagnosis.
