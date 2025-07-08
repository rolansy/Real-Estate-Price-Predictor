FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code and artifacts
COPY server/ ./server/
COPY banglore_home_prices_model.pickle .
COPY columns.json .

# Create artifacts directory and copy model files
RUN mkdir -p server/artifacts
COPY banglore_home_prices_model.pickle server/artifacts/
COPY columns.json server/artifacts/

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=server/server.py
ENV FLASK_ENV=production

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server.server:app"] 