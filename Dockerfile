FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/
COPY app.py .
COPY data/ ./data/

# Install dependencies
RUN uv pip install --system -e .

# Create output directory
RUN mkdir -p output

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
