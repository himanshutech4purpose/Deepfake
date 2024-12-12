
# Use an official Python 3.7 image as the base
FROM python:3.7-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y wget && apt-get clean
# Install build tools
RUN apt-get update && apt-get install -y gcc python3-dev libffi-dev

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run when the container starts
# CMD ["python3", "app.py"]

CMD ["python3", "train.py", "-c", "configs/ljs_base.json", "-m", "ljs_base"]
