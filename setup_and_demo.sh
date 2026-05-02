#!/bin/bash
# Complete Math RAG Setup & Demo Flow

set -e

COLOR_GREEN='\033[0;32m'
COLOR_BLUE='\033[0;34m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${COLOR_BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${COLOR_BLUE}║  Math Learning RAG - Complete Setup   ║${NC}"
echo -e "${COLOR_BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Activate venv
if [ ! -d ".venv" ]; then
    echo -e "${COLOR_RED}❌ Virtual environment not found${NC}"
    echo "   Create one first: python3 -m venv .venv"
    exit 1
fi

source .venv/bin/activate
echo -e "${COLOR_GREEN}✓${NC} Virtual environment activated"

# Step 2: Install dependencies
echo ""
echo -e "${COLOR_BLUE}📦 Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Try installing Nougat with pre-built wheels
echo -n "   Installing OCR tools... "
if pip install -q --only-binary :all: nougat-ocr pillow 2>/dev/null; then
    echo -e "${COLOR_GREEN}✓${NC} Nougat"
else
    echo -e "${COLOR_YELLOW}⚠${NC} Nougat (will use fallback)"
    pip install -q pytesseract ocrmypdf pillow 2>/dev/null || true
fi

# Ensure math_doc_processor dependencies
pip install -q python-dotenv > /dev/null 2>&1

echo -e "${COLOR_GREEN}✓${NC} Dependencies installed"

# Step 3: Create directories
echo ""
echo -e "${COLOR_BLUE}📁 Creating directories...${NC}"
mkdir -p ocr_input ocr_output math_docs_processed internal_docs
echo -e "${COLOR_GREEN}✓${NC} Directories ready"

# Step 4: Create sample documents if ocr_input is empty
echo ""
echo -e "${COLOR_BLUE}📚 Preparing sample math documents...${NC}"
if [ ! "$(ls -A ocr_input 2>/dev/null)" ]; then
    echo "   ocr_input/ is empty. Creating samples..."
    python3 create_sample_math_docs.py
    echo -e "${COLOR_GREEN}✓${NC} Sample documents created"
else
    FILE_COUNT=$(find ocr_input -maxdepth 1 -type f | wc -l)
    echo -e "${COLOR_GREEN}✓${NC} Found $FILE_COUNT file(s) in ocr_input/"
fi

# Step 5: Process documents
echo ""
echo -e "${COLOR_BLUE}🔧 Processing documents...${NC}"
echo "   This may take a moment..."
if ./process_math_docs.sh; then
    echo -e "${COLOR_GREEN}✓${NC} Documents processed"
else
    echo -e "${COLOR_YELLOW}⚠${NC} Processing had issues, but continuing..."
fi

# Step 6: Verify internal_docs
echo ""
echo -e "${COLOR_BLUE}📋 Checking ingestion status...${NC}"
CHUNK_COUNT=$(find internal_docs -name "*.jsonl" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
if [ "$CHUNK_COUNT" -gt 0 ]; then
    echo -e "${COLOR_GREEN}✓${NC} Found $CHUNK_COUNT chunks ready for ingestion"
else
    echo -e "${COLOR_YELLOW}⚠${NC} No chunks found yet. Will ingest during demo."
fi

# Step 7: Ready for demo
echo ""
echo -e "${COLOR_BLUE}════════════════════════════════════════${NC}"
echo -e "${COLOR_GREEN}✅ Setup Complete!${NC}"
echo -e "${COLOR_BLUE}════════════════════════════════════════${NC}"
echo ""
echo "📝 Next steps:"
echo ""
echo -e "  ${COLOR_YELLOW}Option A: Run Interactive Demo${NC}"
echo "    python demo_math_rag.py"
echo ""
echo -e "  ${COLOR_YELLOW}Option B: Use WebUI${NC}"
echo "    lightrag-server"
echo "    Then open: ${COLOR_BLUE}http://localhost:8000${NC}"
echo ""
echo -e "  ${COLOR_YELLOW}Option C: Add Your Own PDFs${NC}"
echo "    1. Copy PDFs to: ocr_input/"
echo "    2. Run: ./process_math_docs.sh"
echo "    3. Run: python demo_math_rag.py"
echo ""
