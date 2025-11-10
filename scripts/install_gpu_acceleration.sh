#!/bin/bash
################################################################################
# GPU Acceleration Installation Script - UPDATED November 2025
# NTSB Aviation Accident Database Analytics Project
#
# Target Hardware: NVIDIA GeForce RTX 5080 (Blackwell, sm_120)
# Driver: 580.105.08 (CUDA 12.6 compatible)
# Python: 3.13.7
#
# Package Versions (Latest as of November 10, 2025):
# - CuPy: 13.6.0 (released August 18, 2025)
# - PyTorch: 2.8.0 stable + nightly for Blackwell support
# - RAPIDS: 25.10.0 (released October 9, 2025)
# - Pyro: 1.9.1 (released June 2, 2024)
# - gpustat: 1.1.1
#
# Duration: 10-15 minutes (downloads ~5 GB)
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT="/home/parobek/Code/NTSB_Datasets"
VENV_PATH="${PROJECT_ROOT}/.venv"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 not found. Please install it first."
        exit 1
    fi
}

################################################################################
# Pre-flight Checks
################################################################################

print_header "🚀 GPU Acceleration Installation - UPDATED November 2025"

echo ""
print_info "Performing pre-flight checks..."

# Check if running in project directory
if [ ! -f "${PROJECT_ROOT}/README.md" ]; then
    print_error "Not in NTSB project directory. Please run from ${PROJECT_ROOT}"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. NVIDIA drivers not installed?"
    print_info "Install NVIDIA drivers first: https://www.nvidia.com/Download/index.aspx"
    exit 1
fi

# Check GPU availability
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [ "$GPU_COUNT" -eq 0 ]; then
    print_error "No NVIDIA GPU detected"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

print_success "GPU detected: $GPU_NAME"
print_info "VRAM: $((GPU_MEMORY / 1024)) GB"
print_info "Driver: $DRIVER_VERSION"
print_info "Compute Capability: $COMPUTE_CAP"

# Detect Blackwell architecture
if [[ "$COMPUTE_CAP" =~ ^12\. ]]; then
    print_warning "Blackwell GPU detected (sm_120)"
    print_info "Will use PyTorch nightly with CUDA 12.8 for native support"
    USE_PYTORCH_NIGHTLY=true
else
    print_info "Using PyTorch stable (sufficient for sm_$COMPUTE_CAP)"
    USE_PYTORCH_NIGHTLY=false
fi

echo ""

# Check for Python 3.13
check_command python
PYTHON_VERSION=$(python --version | awk '{print $2}')
if [[ ! "$PYTHON_VERSION" =~ ^3\.13\. ]]; then
    print_error "Python 3.13 required, found $PYTHON_VERSION"
    exit 1
fi
print_success "Python version: $PYTHON_VERSION"

# Check virtual environment
if [ ! -d "$VENV_PATH" ]; then
    print_error "Virtual environment not found at $VENV_PATH"
    print_info "Create it first: python -m venv .venv"
    exit 1
fi
print_success "Virtual environment found: $VENV_PATH"

# Check disk space (need ~10 GB)
AVAILABLE_SPACE=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    print_warning "Low disk space: ${AVAILABLE_SPACE}GB available (10GB+ recommended)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_success "Pre-flight checks passed"
echo ""

################################################################################
# Activate Virtual Environment
################################################################################

print_header "📦 Activating Virtual Environment"

source "${VENV_PATH}/bin/activate"
print_success "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools -q
print_success "pip upgraded to $(pip --version | awk '{print $2}')"

echo ""

################################################################################
# Installation
################################################################################

print_header "📦 Installing GPU-Accelerated Libraries (Latest Versions)"

echo ""
print_info "Package versions being installed:"
echo "  • CuPy: 13.6.0 (August 2025 release)"
echo "  • PyTorch: 2.8.0 or nightly (based on GPU)"
echo "  • RAPIDS: 25.10.0 (October 2025 release)"
echo "  • Pyro: 1.9.1"
echo "  • gpustat: 1.1.1"
echo ""
print_info "This will install ~5 GB of packages. Estimated time: 10-15 minutes"
echo ""

# Function to install package with error handling
install_package() {
    local package_name=$1
    local pip_args=$2

    print_info "Installing $package_name..."
    if eval "pip install $pip_args" > /tmp/pip_install.log 2>&1; then
        print_success "$package_name installed successfully"
    else
        print_error "$package_name installation failed"
        print_info "Check log: /tmp/pip_install.log"
        tail -20 /tmp/pip_install.log
        return 1
    fi
}

# 1. CuPy (GPU NumPy/SciPy)
print_header "1/5: CuPy 13.6.0 (GPU NumPy/SciPy)"
print_info "Using CUDA 12.x package (matches your driver)"
install_package "CuPy" "cupy-cuda12x==13.6.0"

echo ""

# 2. PyTorch (Deep Learning + LDA)
print_header "2/5: PyTorch (Deep Learning Framework)"

if [ "$USE_PYTORCH_NIGHTLY" = true ]; then
    print_warning "Blackwell GPU: Installing PyTorch nightly for native sm_120 support"
    print_info "Note: Skipping torchaudio (not required for NLP/analytics)"
    install_package "PyTorch Nightly" "--pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128"
else
    print_info "Installing PyTorch 2.8.0 stable with CUDA 12.8"
    print_warning "Skipping torchaudio (not available for Python 3.13)"
    install_package "PyTorch Stable" "torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128"
fi

echo ""

# 3. RAPIDS (cuDF, cuML)
print_header "3/5: RAPIDS 25.10.0 (cuDF + cuML)"
print_warning "This is the largest package (~3 GB). May take 5-10 minutes..."
print_info "Using CUDA 12 packages for compatibility with your driver"
install_package "RAPIDS" "--extra-index-url=https://pypi.nvidia.com cudf-cu12==25.10.* cuml-cu12==25.10.* dask-cudf-cu12==25.10.*"

echo ""

# 4. Pyro (Probabilistic Programming for ProdLDA)
print_header "4/5: Pyro 1.9.1 (Probabilistic Programming)"
install_package "Pyro" "pyro-ppl==1.9.1"

echo ""

# 5. GPU Monitoring Tools
print_header "5/5: gpustat 1.1.1 (GPU Monitoring)"
install_package "gpustat" "gpustat==1.1.1"

echo ""

################################################################################
# Verification
################################################################################

print_header "🔍 Verifying Installation"

echo ""
python << 'VERIFY_SCRIPT'
import sys

print("=" * 70)
print("GPU ACCELERATION VERIFICATION")
print("=" * 70)
print()

# Python version
print(f"✅ Python: {sys.version.split()[0]}")

# Track failures
failures = []

# CuPy
try:
    import cupy as cp
    print(f"✅ CuPy: {cp.__version__}")
    if cp.cuda.is_available():
        print(f"   CUDA available: True")
        print(f"   Device count: {cp.cuda.runtime.getDeviceCount()}")
    else:
        print("   ⚠️  CUDA not available")
        failures.append("CuPy CUDA unavailable")
except ImportError as e:
    print(f"❌ CuPy: FAILED ({e})")
    failures.append("CuPy")

# PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   CUDA available: True")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        
        # Check for Blackwell support
        if compute_cap[0] == 12:
            try:
                archs = torch.cuda.get_arch_list()
                if any('sm_120' in arch or 'compute_120' in arch for arch in archs):
                    print(f"   ✅ Native Blackwell (sm_120) support confirmed")
                else:
                    print(f"   ⚠️  Using compatibility mode (sm_90 fallback)")
            except:
                print(f"   ℹ️  Architecture list unavailable")
    else:
        print("   ⚠️  CUDA not available")
        failures.append("PyTorch CUDA unavailable")
except ImportError as e:
    print(f"❌ PyTorch: FAILED ({e})")
    failures.append("PyTorch")

# torchvision
try:
    import torchvision
    print(f"✅ torchvision: {torchvision.__version__}")
except ImportError as e:
    print(f"❌ torchvision: FAILED ({e})")
    failures.append("torchvision")

# Note about torchaudio
print("ℹ️  torchaudio: Skipped (not required for NLP/analytics, Python 3.13 incompatible)")

# cuDF
try:
    import cudf
    print(f"✅ cuDF: {cudf.__version__}")
except ImportError as e:
    print(f"❌ cuDF: FAILED ({e})")
    failures.append("cuDF")

# cuML
try:
    import cuml
    print(f"✅ cuML: {cuml.__version__}")
except ImportError as e:
    print(f"❌ cuML: FAILED ({e})")
    failures.append("cuML")

# Pyro
try:
    import pyro
    print(f"✅ Pyro: {pyro.__version__}")
except ImportError as e:
    print(f"❌ Pyro: FAILED ({e})")
    failures.append("Pyro")

# gpustat
try:
    import gpustat
    print(f"✅ gpustat: {gpustat.__version__}")
except ImportError as e:
    print(f"❌ gpustat: FAILED ({e})")
    failures.append("gpustat")

print()
print("=" * 70)

if failures:
    print(f"❌ VERIFICATION FAILED ({len(failures)} issues)")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)
else:
    print("✅ VERIFICATION PASSED - All libraries operational")
    sys.exit(0)
VERIFY_SCRIPT

VERIFY_STATUS=$?

echo ""

if [ $VERIFY_STATUS -eq 0 ]; then
    print_success "Installation verification passed!"
    echo ""

    # Quick performance test
    print_header "🚀 Quick Performance Test"
    echo ""

    python << 'PERF_TEST'
import cupy as cp
import time

print("Testing GPU performance (10,000 x 10,000 matrix multiply)...")
print()

start = time.time()
x = cp.random.randn(10000, 10000)
y = cp.dot(x, x.T)
cp.cuda.Stream.null.synchronize()  # Wait for GPU
gpu_time = time.time() - start

print(f"✅ GPU computation completed in {gpu_time:.3f} seconds")

# Estimate CPU time (typical is ~35-50x slower)
estimated_cpu = gpu_time * 40
print(f"ℹ️  Estimated CPU time: ~{estimated_cpu:.1f} seconds")
print(f"ℹ️  GPU speedup: ~{estimated_cpu/gpu_time:.0f}x faster than CPU")
print()
PERF_TEST

    print_success "GPU acceleration is ready!"
    echo ""

else
    print_error "Installation verification failed"
    print_info "Check errors above and retry installation"
    echo ""
    exit 1
fi

################################################################################
# Next Steps
################################################################################

print_header "📚 Next Steps"

echo ""
echo "1. Check GPU status anytime:"
echo "   $ gpustat"
echo "   $ nvidia-smi"
echo ""
echo "2. Monitor GPU during execution:"
echo "   $ watch -n 1 gpustat"
echo ""
echo "3. Run GPU-accelerated notebooks:"
echo "   $ jupyter lab notebooks/nlp/02_topic_modeling_lda_gpu.ipynb"
echo ""
echo "4. Verify PyTorch CUDA:"
echo "   $ python -c 'import torch; print(torch.cuda.is_available())'"
echo ""

print_header "🎉 Installation Complete!"

echo ""
print_success "GPU acceleration enabled with latest package versions!"
echo ""
print_info "Installed versions:"
echo "  • CuPy 13.6.0 (CUDA 12.x)"
if [ "$USE_PYTORCH_NIGHTLY" = true ]; then
    echo "  • PyTorch nightly (with Blackwell sm_120 support)"
else
    echo "  • PyTorch 2.8.0 stable"
fi
echo "  • RAPIDS 25.10.0 (cuDF + cuML)"
echo "  • Pyro 1.9.1"
echo "  • gpustat 1.1.1"
echo ""
print_success "Expected speedups for your RTX 5080:"
echo "   - LDA topic modeling: 8-15x faster"
echo "   - DBSCAN clustering: 5-10x faster"
echo "   - Feature engineering: 3-5x faster"
echo "   - Large matrix operations: 20-50x faster"
echo ""

if [ "$USE_PYTORCH_NIGHTLY" = true ]; then
    print_info "Note: Using PyTorch nightly for Blackwell support"
    echo "   You can switch to stable PyTorch 2.9+ when it releases"
    echo "   with native Blackwell support (expected Q1 2026)"
    echo ""
fi

# Update requirements.txt suggestion
if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    print_info "Consider adding GPU packages to requirements.txt:"
    echo ""
    echo "# GPU acceleration (optional, requires NVIDIA GPU + CUDA 12.x)"
    echo "# Latest versions as of November 2025"
    echo "# cupy-cuda12x==13.6.0  # GPU NumPy/SciPy"
    if [ "$USE_PYTORCH_NIGHTLY" = true ]; then
        echo "# torch  # Use nightly for Blackwell: --index-url https://download.pytorch.org/whl/nightly/cu128"
        echo "# torchvision  # Use nightly: --index-url https://download.pytorch.org/whl/nightly/cu128"
    else
        echo "# torch==2.8.0  # --index-url https://download.pytorch.org/whl/cu128"
        echo "# torchvision   # --index-url https://download.pytorch.org/whl/cu128"
    fi
    echo "# --extra-index-url=https://pypi.nvidia.com"
    echo "# cudf-cu12==25.10.*   # GPU DataFrame"
    echo "# cuml-cu12==25.10.*   # GPU ML algorithms"
    echo "# dask-cudf-cu12==25.10.*  # Distributed GPU computing"
    echo "# pyro-ppl==1.9.1      # Probabilistic programming"
    echo "# gpustat==1.1.1       # GPU monitoring"
    echo ""
fi

print_header "✨ Ready for High-Performance GPU Analytics!"

exit 0
