FROM --platform=linux/amd64 python:3

# Update package lists and install development essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    nano \
    nodejs \
    npm \
    openjdk-17-jdk \
    maven \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir \
    --upgrade pip \
    setuptools \
    wheel \
    uv

RUN uv pip install --no-cache-dir \
    jupyterlab \
    pyspark \
    pandas \
    numpy \
    scikit-learn \
    lightgbm \
    --system

# Create workspace directory
RUN mkdir -p /app

# Set working directory
WORKDIR /app

# Expose JupyterLab port
EXPOSE 8888

# Start JupyterLab server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]