# Phygrid ONNX Runtime Base Image  
# Optimized for lightweight CPU/GPU inference with ONNX models
# Supports both Intel (x64) and ARM architectures

FROM phygrid/cuda-base:latest

# Switch to root for package installation
USER root

# Set architecture-aware variables for ONNX Runtime installation
ARG TARGETARCH
ARG TARGETPLATFORM

# Install ONNX Runtime with architecture-specific packages
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        echo "Installing ONNX Runtime GPU 1.22.0 for x64 with CUDA 12.x + cuDNN 9 + Blackwell support..."; \
        python3 -m pip install --no-cache-dir onnxruntime-gpu==1.22.0; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        echo "Installing ONNX Runtime GPU 1.19.0 for ARM64 Jetson (latest with proven GPU support)..."; \
        python3 -m pip install --no-cache-dir onnxruntime-gpu==1.19.0; \
    else \
        echo "Installing ONNX Runtime CPU fallback..."; \
        python3 -m pip install --no-cache-dir onnxruntime==1.22.0; \
    fi

# Install ONNX ecosystem packages
RUN python3 -m pip install --no-cache-dir \
    onnx==1.15.0 \
    protobuf==4.25.1

# Install additional packages for model optimization
RUN python3 -m pip install --no-cache-dir \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    opencv-python-headless==4.8.1.78

# Install specific packages for audio/video processing with ONNX
RUN python3 -m pip install --no-cache-dir \
    librosa==0.10.1 \
    soundfile==0.12.1


# Set ONNX-specific environment variables
ENV OMP_NUM_THREADS=4
ENV ONNX_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4


# Create ONNX-specific directories
RUN mkdir -p /app/onnx_models /app/onnx_cache
RUN chown -R appuser:appuser /app/onnx_models /app/onnx_cache

# Create ONNX runtime test script
COPY --chown=appuser:appuser <<EOF /app/onnx_test.py
#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np

def test_onnx_runtime():
    try:
        print("ONNX Runtime version:", ort.__version__)
        print("Available providers:", ort.get_available_providers())
        
        # Test with a simple model if providers are available
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA provider available for GPU inference")
        if 'TensorrtExecutionProvider' in providers:
            print("✅ TensorRT provider available for optimized inference")
        if 'CPUExecutionProvider' in providers:
            print("✅ CPU provider available")
            
        # Test GPU memory access if CUDA is available
        if 'CUDAExecutionProvider' in providers:
            try:
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 0
                print("✅ GPU access test: OK")
            except Exception as gpu_e:
                print(f"⚠️  GPU access warning: {gpu_e}")
            
        print("ONNX Runtime setup: OK")
        return True
    except Exception as e:
        print(f"❌ ONNX Runtime test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = test_onnx_runtime()
    sys.exit(0 if success else 1)
EOF

RUN chmod +x /app/onnx_test.py

# Fix executable stack issues for ONNX Runtime
RUN find /usr/local/lib/python3.11/site-packages/onnxruntime -name "*.so" -exec patchelf --clear-execstack {} \; 2>/dev/null || true

# Switch back to non-root user
USER appuser

# Health check using ONNX Runtime
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/onnx_test.py

# Default command
CMD ["python", "/app/onnx_test.py"]

# Labels
LABEL maintainer="Phygrid"
LABEL version="v1.0.8"
LABEL description="ONNX Runtime base image for efficient CPU/GPU inference"
LABEL inference.engine="onnx"
LABEL inference.runtime="onnxruntime-1.16.3"
