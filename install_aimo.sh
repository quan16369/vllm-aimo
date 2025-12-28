#!/bin/bash
# AIMO vLLM Custom Build Script
# Installs vLLM with custom AIMO optimizations

set -e

echo "=========================================="
echo "AIMO vLLM Custom Optimization Build"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$SCRIPT_DIR"

echo -e "${YELLOW}[1/5] Checking Python environment...${NC}"
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

if ! python -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo -e "${RED}Error: Python 3.8+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

echo -e "${YELLOW}[2/5] Checking CUDA availability...${NC}"
if python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null | grep -q "True"; then
    cuda_version=$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo "unknown")
    echo "CUDA available: Yes (version $cuda_version)"
    echo -e "${GREEN}✓ CUDA OK${NC}"
else
    echo -e "${RED}Warning: CUDA not available. vLLM requires CUDA.${NC}"
fi
echo ""

echo -e "${YELLOW}[3/5] Listing custom AIMO files...${NC}"
echo "Custom files added:"
custom_files=(
    "vllm/aimo_integration.py"
    "vllm/aimo_sampling.py"
    "vllm/attention/aimo_math_cache.py"
    "vllm/core/aimo_scheduler.py"
    "AIMO_MODIFICATIONS.md"
)

for file in "${custom_files[@]}"; do
    if [ -f "$VLLM_DIR/$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file (missing!)"
    fi
done
echo ""

echo -e "${YELLOW}[4/5] Installing vLLM with custom optimizations...${NC}"
echo "This may take 5-10 minutes..."

cd "$VLLM_DIR"

# Backup existing installation (if any)
if python -c 'import vllm' 2>/dev/null; then
    echo "Backing up existing vLLM installation..."
    pip uninstall -y vllm 2>/dev/null || true
fi

# Install in editable mode
echo "Installing vLLM (editable mode)..."
pip install -e . --no-build-isolation 2>&1 | tee install.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ vLLM installed successfully${NC}"
else
    echo -e "${RED}✗ Installation failed. Check install.log for details.${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}[5/5] Verifying installation...${NC}"

# Test import
if python -c 'import vllm; print("vLLM version:", vllm.__version__)' 2>/dev/null; then
    echo -e "${GREEN}✓ vLLM import successful${NC}"
else
    echo -e "${RED}✗ vLLM import failed${NC}"
    exit 1
fi

# Test custom modules
echo "Testing custom AIMO modules:"

test_imports=(
    "from vllm.aimo_integration import enable_aimo_optimizations"
    "from vllm.aimo_sampling import create_aimo_sampler"
    "from vllm.attention.aimo_math_cache import get_math_cache"
    "from vllm.core.aimo_scheduler import create_aimo_scheduler"
)

all_ok=true
for import_stmt in "${test_imports[@]}"; do
    module_name=$(echo "$import_stmt" | sed 's/from vllm\.\([^ ]*\).*/\1/' | tr '.' '/')
    if python -c "$import_stmt" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $module_name"
    else
        echo -e "  ${RED}✗${NC} $module_name (import failed)"
        all_ok=false
    fi
done

echo ""
echo "=========================================="
if [ "$all_ok" = true ]; then
    echo -e "${GREEN}✓ Installation complete!${NC}"
    echo ""
    echo "AIMO optimizations are now active."
    echo ""
    echo "Next steps:"
    echo "1. In your notebook, add:"
    echo "   from vllm.aimo_integration import enable_aimo_optimizations"
    echo "   enable_aimo_optimizations(K=4)"
    echo ""
    echo "2. Start vLLM server with optimized args"
    echo "3. Run your AIMO inference"
    echo ""
    echo "Documentation: $VLLM_DIR/AIMO_MODIFICATIONS.md"
else
    echo -e "${RED}✗ Some imports failed${NC}"
    echo "Check the error messages above"
fi
echo "=========================================="
