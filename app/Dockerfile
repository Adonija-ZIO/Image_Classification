FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set up working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Start the app
CMD ["streamlit", "run", "main.py"]
