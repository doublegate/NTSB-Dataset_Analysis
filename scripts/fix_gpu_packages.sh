#!/bin/bash
################################################################################
# GPU Package Fix Script - Remove Old Versions First
# Fixes the issue where old packages remain after installation
################################################################################

set -e
set -u

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Check venv
if [ -z "${VIRTUAL_ENV:-}" ]; then
    print_error "Not in virtual environment!"
    echo "Run: source ~/Code/NTSB_Datasets/.venv/bin/activate"
    exit 1
fi

print_header "🔧 GPU Package Fix - Remove Old Versions"
echo ""

################################################################################
# Step 1: Remove Old CuPy
################################################################################

print_header "Step 1/3: Removing Old CuPy Packages"
echo ""

print_info "Checking for cupy-cuda13x..."
if pip show cupy-cuda13x &>/dev/null; then
    print_warning "Found cupy-cuda13x (wrong version)"
    pip uninstall cupy-cuda13x -y
    print_success "Removed cupy-cuda13x"
else
    print_info "cupy-cuda13x not installed (good)"
fi

print_info "Checking for cupy-cuda11x..."
if pip show cupy-cuda11x &>/dev/null; then
    print_warning "Found cupy-cuda11x"
    pip uninstall cupy-cuda11x -y
    print_success "Removed cupy-cuda11x"
fi

print_info "Installing cupy-cuda12x..."
pip install cupy-cuda12x==13.6.0 --no-cache-dir
print_success "Installed cupy-cuda12x 13.6.0"

echo ""

################################################################################
# Step 2: Remove Old PyTorch
################################################################################

print_header "Step 2/3: Removing Old PyTorch Packages"
echo ""

print_info "Checking for old PyTorch versions..."
if pip show torch &>/dev/null; then
    CURRENT_TORCH=$(pip show torch | grep "Version:" | awk '{print $2}')
    print_warning "Found PyTorch $CURRENT_TORCH"
    pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
    print_success "Removed old PyTorch"
else
    print_info "PyTorch not installed"
fi

print_info "Installing PyTorch nightly for Blackwell..."
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
print_success "Installed PyTorch nightly"

echo ""

################################################################################
# Step 3: Verify RAPIDS
################################################################################

print_header "Step 3/3: Verifying RAPIDS"
echo ""

print_info "Checking RAPIDS versions..."
if pip show cudf-cu12 &>/dev/null; then
    CUDF_VERSION=$(pip show cudf-cu12 | grep "Version:" | awk '{print $2}')
    print_success "cuDF: $CUDF_VERSION"
else
    print_warning "cuDF not found, installing..."
    pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==25.10.* --no-cache-dir
fi

if pip show cuml-cu12 &>/dev/null; then
    CUML_VERSION=$(pip show cuml-cu12 | grep "Version:" | awk '{print $2}')
    print_success "cuML: $CUML_VERSION"
else
    print_warning "cuML not found, installing..."
    pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==25.10.* --no-cache-dir
fi

echo ""

################################################################################
# Verification
################################################################################

print_header "🔍 Verification"
echo ""

python << 'VERIFY'
import sys

print("=" * 70)
print("VERIFICATION AFTER FIX")
print("=" * 70)
print()

failures = []

# CuPy
try:
    import cupy as cp
    print(f"✅ CuPy: {cp.__version__}")
    if cp.cuda.is_available():
        print(f"   CUDA available: True")
        # Test actual GPU operation
        x = cp.array([1, 2, 3])
        y = cp.sum(x)
        print(f"   Test operation: Success (sum={int(y)})")
    else:
        print("   ❌ CUDA not available")
        failures.append("CuPy CUDA")
except Exception as e:
    print(f"❌ CuPy: FAILED - {e}")
    failures.append("CuPy")

print()

# PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   CUDA available: True")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"   Compute Capability: {cap[0]}.{cap[1]}")
        
        # Check for sm_120 support
        if cap[0] == 12:
            try:
                archs = torch.cuda.get_arch_list()
                has_sm120 = any('sm_120' in a or 'compute_120' in a for a in archs)
                if has_sm120:
                    print(f"   ✅ Native Blackwell sm_120 support!")
                else:
                    print(f"   ⚠️  sm_90 fallback mode")
                    print(f"   Supported: {archs}")
            except:
                pass
        
        # Test GPU operation
        x = torch.randn(100, 100, device='cuda')
        y = torch.matmul(x, x.T)
        torch.cuda.synchronize()
        print(f"   Test operation: Success")
    else:
        print("   ❌ CUDA not available")
        failures.append("PyTorch CUDA")
except Exception as e:
    print(f"❌ PyTorch: FAILED - {e}")
    failures.append("PyTorch")

print()

# cuDF
try:
    import cudf
    print(f"✅ cuDF: {cudf.__version__}")
except Exception as e:
    print(f"⚠️  cuDF: {e}")

# cuML
try:
    import cuml
    print(f"✅ cuML: {cuml.__version__}")
except Exception as e:
    print(f"⚠️  cuML: {e}")

print()
print("=" * 70)

if failures:
    print(f"❌ ISSUES DETECTED: {', '.join(failures)}")
    sys.exit(1)
else:
    print("✅ ALL CRITICAL PACKAGES WORKING!")
    sys.exit(0)
VERIFY

VERIFY_STATUS=$?

echo ""

if [ $VERIFY_STATUS -eq 0 ]; then
    print_header "🎉 Fix Complete!"
    echo ""
    print_success "GPU acceleration is now properly configured!"
    echo ""
    print_info "Test GPU performance:"
    echo "  python -c 'import cupy as cp; import time; start=time.time(); x=cp.random.randn(10000,10000); y=cp.dot(x,x.T); cp.cuda.Stream.null.synchronize(); print(f\"Time: {time.time()-start:.3f}s\")'"
    echo ""
else
    print_header "⚠️  Fix Incomplete"
    echo ""
    print_warning "Some packages still have issues. Check output above."
    echo ""
fi

exit $VERIFY_STATUS
