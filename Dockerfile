# ==============================================================================
# Dockerfile: Unified Multi-Backend Environment for MEC Task Offloading (D3QN)
# Supports:
# 1. Legacy Implementation: Python 3.7 + TensorFlow 1.15.5
# 2. Strict Paper-Aligned Implementation: Python 3.7 + PyTorch 1.13.1 + .cfg
# ==============================================================================

FROM python:3.7.9

WORKDIR /app

# Upgrade pip to compatible modern version
RUN pip install --no-cache-dir --upgrade pip==23.0.1

# 1. Install legacy TensorFlow 1.15.5 stack and dependencies
RUN pip install --no-cache-dir \
    numpy==1.18.5 \
    tensorflow==1.15.5 \
    h5py==2.10.0 \
    protobuf==3.20.3 \
    ipykernel==6.16.2 \
    matplotlib==3.5.3

# 2. Install PyTorch CPU (compatible with Python 3.7) for paper_aligned_code
RUN pip install --no-cache-dir torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Set environment variables for smooth Python execution
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command: Menjalankan pipeline eksperimen paper-aligned dengan sudden_shift_reset.cfg
CMD ["python3", "paper_aligned_code/run_experiment.py", "--config", "paper_aligned_code/configs/sudden_shift_reset.cfg"]
