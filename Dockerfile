# Use the official Python 3.7 image
FROM python:3.7-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --allow-unauthenticated \
    gcc libpq-dev ffmpeg

# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN  pip install torch
RUN pip install -r requirements.txt



RUN apt-get install wget -y

# Downloading Model Files
RUN wget https://cdn-api.dev.ks.samagra.io/cm-video-files/wav2lip_gan.pth -P Wav2Lip/checkpoints/
RUN wget https://cdn-api.dev.ks.samagra.io/cm-video-files/s3fd.pth -P Wav2Lip/face_detection/detection/sfd/

# Copy the FastAPI app files to the container
COPY . /app/
# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "main_old:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["python", "test1.py"]