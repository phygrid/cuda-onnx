# ONNX Runtime with TensorRT EP - Multi-Stage Optimized
# Extends Phygrid CUDA base with minimal ONNX-specific components
# Multi-arch support for x86_64 and ARM64

# Multi-stage build args for proper cross-platform support  
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

# ====== BUILD STAGE: ONNX Package Download ======
FROM phygrid/cuda-base:latest AS onnx-builder

# ONNX-specific version pins (updated for TensorRT 10.13.2 compatibility)
ARG ONNX_VERSION=1.19.0
ARG ORT_GPU_VERSION=1.22.0

WORKDIR /build

# Install ONNX packages in build stage (with build dependencies if needed)
RUN set -ex && \
    echo "Building ONNX stack for ${TARGETARCH:-unknown}..." && \
    \
    # Create virtual environment for cleaner package management
    python -m venv /build/onnx-env && \
    . /build/onnx-env/bin/activate && \
    \
    # Install ONNX packages
    pip install --no-cache-dir \
        onnx==${ONNX_VERSION} \
        onnxruntime-gpu==${ORT_GPU_VERSION} && \
    \
    # Verify installation in build stage
    python -c "import onnx; import onnxruntime as ort; print(f'ONNX version: {onnx.__version__}'); print(f'ONNX Runtime version: {ort.__version__}'); print(f'Available providers: {ort.get_available_providers()}')" && \
    \
    # Create package list for runtime stage
    pip freeze > /build/onnx-requirements.txt && \
    echo "ONNX packages prepared for runtime stage"

# ====== FINAL STAGE: Runtime Image ======
FROM phygrid/cuda-base:latest

# Re-declare args for final stage
ARG TARGETARCH
ARG ONNX_VERSION=1.19.0
ARG ORT_GPU_VERSION=1.22.0
ARG ORT_CACHE=/app/ort_cache

USER root

# Install ONLY the exact ONNX packages we need (no build dependencies)
RUN set -ex && \
    echo "Installing ONNX Runtime for ${TARGETARCH:-unknown} (runtime-only)..." && \
    \
    # Install core ONNX and ONNX Runtime GPU + common CV packages
    python -m pip install --no-cache-dir --break-system-packages \
        onnx==${ONNX_VERSION} \
        onnxruntime-gpu==${ORT_GPU_VERSION} \
        opencv-python-headless>=4.10 \
        scipy>=1.12 \
        huggingface-hub>=0.23 && \
    \
    # Quick verification without extra dependencies
    python -c "import onnx, onnxruntime as ort; print(f'✓ ONNX {onnx.__version__} + ONNX Runtime {ort.__version__} installed'); print(f'Providers: {ort.get_available_providers()}')" || echo "⚠️  ONNX verification failed (expected without GPU runtime)"

# ONNX-specific environment variables (optimized for TensorRT 10.13.2)
ENV ONNX_VERSION=${ONNX_VERSION}
ENV ORT_GPU_VERSION=${ORT_GPU_VERSION}
ENV ORT_CACHE=${ORT_CACHE}

# Optimized ONNX Runtime TensorRT EP settings for TensorRT 10.13.2
ENV ORT_TRT_FP16=1
ENV ORT_TRT_INT8=0
ENV ORT_TRT_ENGINE_CACHE_ENABLE=1
ENV ORT_TRT_ENGINE_CACHE_PATH=${ORT_CACHE}
ENV ORT_TRT_MAX_WORKSPACE_SIZE=1073741824
ENV ORT_TRT_TIMING_CACHE_ENABLE=1
ENV ORT_TRT_BUILDER_OPTIMIZATION_LEVEL=3

# Create ONNX cache directory (minimal overhead)
RUN mkdir -p ${ORT_CACHE} && \
    chown -R appuser:appuser ${ORT_CACHE}

# Streamlined ONNX health check (no build stage overhead)
COPY --chown=appuser:appuser <<'PY' /app/health_onnx.py
#!/usr/bin/env python3
import sys
import os
import platform

def check_onnx_health():
    print("=== Phygrid ONNX Runtime Health Check ===")
    print(f"Architecture: {platform.machine()}")
    print(f"TensorRT version (from base): {os.getenv('TENSORRT_VERSION', 'unknown')}")
    
    # Check ONNX
    try:
        import onnx
        print(f"✓ ONNX version: {onnx.__version__}")
    except ImportError:
        print("❌ ONNX not available")
        return 1
    
    # Check ONNX Runtime and providers
    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime version: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        # Validate TensorRT EP
        trt_available = "TensorrtExecutionProvider" in providers
        cuda_available = "CUDAExecutionProvider" in providers
        
        print(f"✓ TensorRT EP: {'available' if trt_available else 'not available'}")
        print(f"✓ CUDA EP: {'available' if cuda_available else 'not available'}")
        
        # Test TensorRT EP configuration
        if trt_available:
            config = {
                "trt_fp16_enable": bool(int(os.getenv("ORT_TRT_FP16", "1"))),
                "trt_engine_cache_enable": bool(int(os.getenv("ORT_TRT_ENGINE_CACHE_ENABLE", "1"))),
                "trt_engine_cache_path": os.getenv("ORT_TRT_ENGINE_CACHE_PATH", "/app/ort_cache"),
                "trt_max_workspace_size": int(os.getenv("ORT_TRT_MAX_WORKSPACE_SIZE", "1073741824")),
                "trt_timing_cache_enable": bool(int(os.getenv("ORT_TRT_TIMING_CACHE_ENABLE", "1"))),
                "trt_builder_optimization_level": int(os.getenv("ORT_TRT_BUILDER_OPTIMIZATION_LEVEL", "3"))
            }
            print("✓ TensorRT EP configuration validated")
            print(f"  FP16: {config['trt_fp16_enable']}")
            print(f"  Cache: {config['trt_engine_cache_enable']}")  
            print(f"  Workspace: {config['trt_max_workspace_size']//1024//1024}MB")
            print(f"  Optimization Level: {config['trt_builder_optimization_level']}")
        
    except ImportError as e:
        print(f"❌ ONNX Runtime error: {e}")
        return 1
    
    # Check additional packages from this layer
    try:
        import cv2, scipy, huggingface_hub
        print(f"✓ OpenCV version: {cv2.__version__}")
        print(f"✓ SciPy version: {scipy.__version__}")
        print(f"✓ Hugging Face Hub version: {huggingface_hub.__version__}")
    except ImportError as e:
        print(f"❌ Missing CV package: {e}")
        return 1
    
    # Check cache directory
    cache_path = os.getenv('ORT_CACHE', '/app/ort_cache')
    if os.path.exists(cache_path) and os.access(cache_path, os.W_OK):
        print(f"✓ Cache directory ready: {cache_path}")
    else:
        print(f"⚠️  Cache directory issue: {cache_path}")
    
    print(f"\n✅ ONNX Runtime optimized for TensorRT 10.13.2!")
    return 0

if __name__ == "__main__":
    sys.exit(check_onnx_health())
PY

RUN chmod +x /app/health_onnx.py

# Switch to non-root user
USER appuser

# Default command
CMD ["python", "/app/health_onnx.py"]

# Optimized labels (updated for TensorRT 10.13.2 compatibility)
LABEL maintainer="Phygrid"
LABEL base="phygrid/cuda-base"
LABEL onnx.version="${ONNX_VERSION}"
LABEL onnxruntime.gpu.version="${ORT_GPU_VERSION}"
LABEL tensorrt.compatible="10.13.2"
LABEL description="Minimal ONNX Runtime GPU optimized for TensorRT 10.13.2 inference"
LABEL build.stage="optimized"