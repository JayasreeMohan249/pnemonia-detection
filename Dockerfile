
# Use a lightweight Python 3.12 base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependency list into the container
COPY requirements.txt .

# Upgrade pip and install application dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application
COPY app.py .

# Copy the trained CNN model
COPY custom_cnn_v2_baseline.keras .

# Copy Streamlit configuration
COPY .streamlit ./.streamlit

# Expose the port used by Streamlit
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
