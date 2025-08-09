# Use Python 3.11 (stable version)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Command to run your application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
