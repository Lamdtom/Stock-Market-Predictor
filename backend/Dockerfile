# Base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into container
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Set environment variable to avoid buffering
ENV PYTHONUNBUFFERED=1

# Command to run FastAPI using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]