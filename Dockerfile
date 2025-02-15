# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install required Python packages
RUN pip install fastapi uvicorn python-multipart requests python-dotenv


# Expose port 8000 for the API
EXPOSE 8000

# Run the application with Uvicorn
CMD ["uvicorn", "llm:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
