# Phygrid CUDA ONNX Image

[![Docker Hub](https://img.shields.io/docker/pulls/phygrid/cuda-onnx.svg)](https://hub.docker.com/r/phygrid/cuda-onnx)
[![Docker Image Version](https://img.shields.io/docker/v/phygrid/cuda-onnx?sort=semver)](https://hub.docker.com/r/phygrid/cuda-onnx/tags)
[![Build Status](https://github.com/phygrid/cuda-onnx/workflows/Build%20and%20Deploy%20Docker%20Image/badge.svg)](https://github.com/phygrid/cuda-onnx/actions)
[![License](https://img.shields.io/github/license/phygrid/cuda-onnx)](LICENSE)

A multi-architecture Docker image optimized for ONNX Runtime inference with GPU acceleration, built on the Phygrid CUDA base image.

## 🚀 Quick Start

```bash
# Pull the latest image
docker pull phygrid/cuda-onnx:latest

# Use as base image in your Dockerfile
FROM phygrid/cuda-onnx:1.0.0
```

## 📋 What's Included

### Base Layer
Built on `phygrid/cuda-base:latest` which includes:
- Python 3.11 with optimized pip, setuptools, wheel
- FastAPI, Uvicorn, Pydantic for web services
- Common system dependencies and security features

### ONNX Specific Additions
- **ONNX Runtime**: Latest version with CPU/GPU execution providers
- **ONNX Ecosystem**: Core ONNX packages and protobuf
- **Audio/Video Processing**: librosa, soundfile for multimedia inference
- **Computer Vision**: OpenCV, scipy, scikit-learn for image processing
- **Model Optimization**: Tools for efficient inference

### Environment Optimizations
- **Threading**: Optimized thread counts for ONNX Runtime
- **Directories**: Pre-created `/app/onnx_models` and `/app/onnx_cache`
- **Security**: Executable stack fixes with patchelf
- **Health Check**: Built-in ONNX Runtime validation

## 🐳 Docker Hub

**Repository**: [phygrid/cuda-onnx](https://hub.docker.com/r/phygrid/cuda-onnx)

### Available Tags
- `latest` - Latest stable release
- `1.0.0`, `1.0.1`, etc. - Specific semantic versions
- Multi-architecture support: `linux/amd64`, `linux/arm64`

## 📦 Usage Examples

### As Base Image
```dockerfile
FROM phygrid/cuda-onnx:1.0.0

# Copy your ONNX models
COPY models/ /app/onnx_models/

# Install additional dependencies
RUN pip install -r requirements.txt

# Override default command
CMD ["python", "inference_server.py"]
```

### Development Environment
```bash
# Run interactive container with GPU support
docker run -it --rm \
  --gpus all \
  -v $(pwd):/app/workspace \
  -v $(pwd)/models:/app/onnx_models \
  -p 8000:8000 \
  phygrid/cuda-onnx:latest \
  bash
```

### Production Deployment
```bash
# Run ONNX inference service
docker run -d \
  --name onnx-inference \
  --gpus all \
  -p 8000:8000 \
  -v /data/models:/app/onnx_models \
  -v /data/cache:/app/onnx_cache \
  -e OMP_NUM_THREADS=8 \
  phygrid/cuda-onnx:latest
```

### Model Inference Example
```python
import onnxruntime as ort
import numpy as np

# Create inference session with GPU support
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('/app/onnx_models/model.onnx', providers=providers)

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run(None, {'input': input_data})
```

## 🏗️ Building from Source

```bash
# Clone repository
git clone https://github.com/phygrid/cuda-onnx.git
cd cuda-onnx

# Build image
docker build -t phygrid/cuda-onnx:custom .

# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t phygrid/cuda-onnx:custom .
```

## 🔄 Versioning

This project uses automated semantic versioning:

- **Automatic**: Patch versions increment on main branch changes
- **Manual**: Edit `VERSION` file for major/minor bumps
- **Tags**: Git tags created automatically (e.g., `v1.0.0`)

## 🧪 Health Check

The image includes a comprehensive health check:

```bash
# Test ONNX Runtime setup
docker run --rm phygrid/cuda-onnx:latest python /app/onnx_test.py

# Expected output:
# ONNX Runtime version: 1.16.3
# Available providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
# ✅ CUDA provider available for GPU inference
# ✅ CPU provider available
# ONNX Runtime setup: OK
```

## ⚙️ Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | `4` | OpenMP thread count |
| `ONNX_NUM_THREADS` | `4` | ONNX Runtime thread count |
| `OPENBLAS_NUM_THREADS` | `4` | OpenBLAS thread count |

### Volume Mounts
| Path | Purpose |
|------|---------|
| `/app/onnx_models` | ONNX model storage |
| `/app/onnx_cache` | Runtime cache directory |
| `/app/data` | Input/output data |
| `/app/logs` | Application logs |

## 🔧 Performance Tuning

### GPU Optimization
```bash
# Enable CUDA provider with specific device
export CUDA_VISIBLE_DEVICES=0

# Configure memory management
export ORT_CUDA_MEMORY_LIMIT=4GB
```

### CPU Optimization
```bash
# Adjust thread counts based on CPU cores
export OMP_NUM_THREADS=$(nproc)
export ONNX_NUM_THREADS=$(nproc)
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone and setup
git clone https://github.com/phygrid/cuda-onnx.git
cd cuda-onnx

# Test build locally
docker build -t phygrid/cuda-onnx:test .
docker run --rm phygrid/cuda-onnx:test python /app/onnx_test.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Labels

The image includes standard OCI labels:

```dockerfile
LABEL org.opencontainers.image.title="Phygrid CUDA ONNX"
LABEL org.opencontainers.image.description="ONNX Runtime base image for efficient CPU/GPU inference"
LABEL org.opencontainers.image.vendor="Phygrid"
LABEL inference.engine="onnx"
LABEL inference.runtime="onnxruntime-1.16.3"
```

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/phygrid/cuda-onnx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/phygrid/cuda-onnx/discussions)
- **Docker Hub**: [phygrid/cuda-onnx](https://hub.docker.com/r/phygrid/cuda-onnx)

## 📈 Metrics

- **Image size**: ~1.2GB compressed
- **Build time**: ~8-15 minutes (with cache)
- **Architectures**: AMD64, ARM64
- **ONNX Runtime version**: 1.16.3
- **Base image**: phygrid/cuda-base:latest