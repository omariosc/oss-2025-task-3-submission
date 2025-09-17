# Multi-stage build for optimized Docker image size
# Stage 1: Builder
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS builder

# Set timezone and avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    ninja-build \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies in builder stage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime (smaller final image)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set timezone and avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages

# Create directories as specified by EndoVis requirements
RUN mkdir -p /app/code /input /output

# Copy the Process_SkillEval.sh script as required by challenge
COPY Process_SkillEval.sh /usr/local/bin/Process_SkillEval.sh
RUN chmod +x /usr/local/bin/Process_SkillEval.sh

# Copy application code - YOLO-Pose System
COPY main_yolo_pose.py /app/code/main.py
# Copy YOLO-Pose model weights
COPY keypoint_training/surgical_keypoints/weights/best.pt /app/code/surgical_keypoints_best.pt
# Copy additional modules if needed
# COPY src/ /app/code/src/

# Set environment variables for GPU support
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV PYTHONPATH=/app/code:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Use Process_SkillEval.sh as entrypoint per challenge requirements
ENTRYPOINT ["/usr/local/bin/Process_SkillEval.sh"]
# Default to TRACK task if no parameter provided
CMD ["TRACK"]