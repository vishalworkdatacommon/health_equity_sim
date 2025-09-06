# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Streamlit and Matplotlib to use writable directories
ENV STREAMLIT_HOME=/app/.streamlit
ENV MPLCONFIGDIR=/app/.config/matplotlib

# Install system dependencies required by lightgbm
RUN apt-get update && apt-get install -y libgomp1

# Create a writable .streamlit directory and a config file to disable telemetry
RUN mkdir -p /app/.streamlit
RUN echo "[global]\nshowWarningOnDirectExecution = false\n[client]\nshowErrorDetails = false\n[browser]\ngatherUsageStats = false\n" > /app/.streamlit/config.toml

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the command to run the app
CMD ["streamlit", "run", "app.py"]