FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . /app

# Ensure project root is on Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
	CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
