# --- Build Stage ---
FROM python:3.13-slim AS builder

# Install system dependencies for psutil, chromadb, and other packages
# chroma-hnswlib requires C++11 compiler support
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install poetry gunicorn

# Copy only dependency-defining files
COPY pyproject.toml ./

# Install dependencies, without dev dependencies, into a virtual environment
RUN poetry config virtualenvs.create false && \
    poetry install --without dev --no-root --no-interaction --no-ansi

# Set a non-root user
RUN addgroup --system app && adduser --system --group app

# Create necessary directories and set permissions
RUN mkdir -p /home/app/app/static/uploads && \
    chown -R app:app /home/app

USER app

# Set working directory
WORKDIR /home/app

COPY ./app ./app
COPY gunicorn_conf.py .
COPY setup_wizard.py .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using our custom Gunicorn config file.
# This ensures structured JSON logging is used in production.
CMD ["gunicorn", "-c", "./gunicorn_conf.py", "app.main:app"]
