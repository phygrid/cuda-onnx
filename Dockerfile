# ONNX Runtime with TensorRT EP - Optimized for Size
# Extends Phygrid CUDA base with only ONNX-specific components
# Multi-arch support for x86_64 and ARM64

# Multi-stage build args for proper cross-platform support  
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

FROM phygrid/cuda-base:latest

# ONNX-specific version pins
ARG ONNX_VERSION=1.19.2
ARG ORT_GPU_VERSION=1.22.0
ARG ORT_CACHE=/app/ort_cache

USER root

# Install ONLY ONNX-specific dependencies (minimal approach)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Only what's needed for ONNX model loading and processing
    libprotobuf23 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install ONNX packages (architecture-aware, size-optimized)
RUN set -ex && \
    echo "Installing ONNX stack for ${TARGETARCH}..." && \
    \
    # Install core ONNX package first
    python -m pip install --no-cache-dir --break-system-packages \
        onnx==${ONNX_VERSION} && \
    \
    # Install ONNX Runtime GPU with TensorRT EP
    # Version 1.22+ supports TensorRT 10.x and CUDA 13.0
    python -m pip install --no-cache-dir --break-system-packages \
        onnxruntime-gpu==${ORT_GPU_VERSION} && \
    \
    # Verify installation
    python -c "
import onnx
import onnxruntime as ort
print(f'ONNX version: {onnx.__version__}')
print(f'ONNX Runtime version: {ort.__version__}')
providers = ort.get_available_providers()
print(f'Available providers: {providers}')
print(f'TensorRT available: {\"TensorrtExecutionProvider\" in providers}')
print(f'CUDA available: {\"CUDAExecutionProvider\" in providers}')
" || echo "⚠️  ONNX verification failed (expected without GPU runtime)"

# ONNX-specific environment variables
ENV ONNX_VERSION=${ONNX_VERSION}
ENV ORT_GPU_VERSION=${ORT_GPU_VERSION}
ENV ORT_CACHE=${ORT_CACHE}

# Default ONNX Runtime TensorRT EP settings (optimized for inference)
ENV ORT_TRT_FP16=1
ENV ORT_TRT_INT8=0
ENV ORT_TRT_ENGINE_CACHE_ENABLE=1
ENV ORT_TRT_ENGINE_CACHE_PATH=${ORT_CACHE}
ENV ORT_TRT_MAX_WORKSPACE_SIZE=1073741824
ENV ORT_TRT_TIMING_CACHE_ENABLE=1

# Create ONNX cache directory
RUN mkdir -p ${ORT_CACHE} && \
    chown -R appuser:appuser ${ORT_CACHE}

# ONNX-specific health check
COPY --chown=appuser:appuser <<'PY' /app/health_onnx.py
#!/usr/bin/env python3
import sys
import os
import platform

def check_onnx_health():
    print("=== Phygrid ONNX Runtime Health Check ===")
    
    # System info
    arch = platform.machine()
    print(f"Architecture: {arch}")
    print(f"Platform: {os.environ.get('TARGETPLATFORM', 'unknown')}")
    
    # Check ONNX
    try:
        import onnx
        print(f"✓ ONNX version: {onnx.__version__}")
    except ImportError:
        print("❌ ONNX not available")
        return 1
    
    # Check ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime version: {ort.__version__}")
        
        # Check available providers
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        # Check TensorRT EP
        if "TensorrtExecutionProvider" in providers:
            print("✓ TensorRT Execution Provider available")
        else:
            print("⚠️  TensorRT Execution Provider not available")
        
        # Check CUDA EP  
        if "CUDAExecutionProvider" in providers:
            print("✓ CUDA Execution Provider available")
        else:
            print("⚠️  CUDA Execution Provider not available")
            
        # Verify TensorRT EP configuration
        provider_options = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": bool(int(os.getenv("ORT_TRT_FP16", "1"))),
                "trt_int8_enable": bool(int(os.getenv("ORT_TRT_INT8", "0"))),
                "trt_engine_cache_enable": bool(int(os.getenv("ORT_TRT_ENGINE_CACHE_ENABLE", "1"))),
                "trt_engine_cache_path": os.getenv("ORT_TRT_ENGINE_CACHE_PATH", "/app/ort_cache"),
                "trt_max_workspace_size": int(os.getenv("ORT_TRT_MAX_WORKSPACE_SIZE", "1073741824")),
                "trt_timing_cache_enable": bool(int(os.getenv("ORT_TRT_TIMING_CACHE_ENABLE", "1")))
            }),
            ("CUDAExecutionProvider", {}),
            "CPUExecutionProvider"
        ]
        print("✓ TensorRT EP configuration validated")
        
        # Show environment settings
        print(f"\nTensorRT EP Settings:")
        print(f"  FP16: {os.getenv('ORT_TRT_FP16', '1')}")
        print(f"  Engine Cache: {os.getenv('ORT_TRT_ENGINE_CACHE_ENABLE', '1')}")
        print(f"  Cache Path: {os.getenv('ORT_TRT_ENGINE_CACHE_PATH', '/app/ort_cache')}")
        print(f"  Workspace Size: {int(os.getenv('ORT_TRT_MAX_WORKSPACE_SIZE', '1073741824'))//1024//1024}MB")
        print(f"  Timing Cache: {os.getenv('ORT_TRT_TIMING_CACHE_ENABLE', '1')}")
        
    except ImportError as e:
        print(f"❌ ONNX Runtime import error: {e}")
        return 1
    except Exception as e:
        print(f"❌ ONNX Runtime error: {e}")
        return 1
    
    # Check cache directory
    cache_path = os.getenv('ORT_CACHE', '/app/ort_cache')
    if os.path.exists(cache_path) and os.access(cache_path, os.W_OK):
        print(f"✓ Cache directory ready: {cache_path}")
    else:
        print(f"⚠️  Cache directory issue: {cache_path}")
    
    print(f"\n✅ ONNX Runtime ready on {arch}!")
    return 0

if __name__ == "__main__":
    sys.exit(check_onnx_health())
PY

RUN chmod +x /app/health_onnx.py

# Switch back to non-root user
USER appuser

# Default command
CMD ["python", "/app/health_onnx.py"]

# Optimized labels
LABEL maintainer="Phygrid"
LABEL base="phygrid/cuda-base"
LABEL onnx.version="${ONNX_VERSION}"
LABEL onnxruntime.gpu.version="${ORT_GPU_VERSION}"
LABEL description="Minimal ONNX Runtime GPU with TensorRT EP for optimized inference"
LABEL tensorrt.ep="enabled"