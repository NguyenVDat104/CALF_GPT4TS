FROM python:3.10-slim

WORKDIR /app

# Setup non-root user (Hugging Face Spaces UID 1000 standard)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Install dependencies
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY --chown=user . /app

# Ensure proper permissions
RUN chmod -R 777 /app/datasets

USER user

# Hugging Face default port
EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
