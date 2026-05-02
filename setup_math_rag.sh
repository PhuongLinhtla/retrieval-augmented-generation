#!/bin/bash
# Setup script for Math RAG with Nougat OCR

set -e

echo "📚 Setting up Math RAG with Nougat OCR..."

# 1. Activate venv if exists
if [ -d ".venv" ]; then
    echo "✓ Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠ No venv found. Create one first: python3 -m venv .venv"
    exit 1
fi

# 2. Install Nougat
echo "🔧 Installing Nougat (Meta's scientific document OCR)..."
# Use pre-built wheels to avoid pyarrow build issues on Python 3.12
pip install --upgrade pip setuptools wheel
pip install -q --only-binary :all: nougat-ocr pillow

# Verify installation
if command -v nougat &> /dev/null; then
    echo "✓ Nougat installed successfully"
    nougat --version
else
    echo "⚠️  Nougat not in PATH, but may still work. Trying alternative check..."
    python3 -c "import nougat; print('✓ Nougat Python module installed')" || {
        echo "❌ Nougat installation failed (pyarrow build issue)"
        echo "⚠️  Falling back to pytesseract + ocrmypdf (lighter alternative)..."
        pip install -q pytesseract ocrmypdf
    }
fi

# 3. Create directories
mkdir -p ocr_input ocr_output math_docs_processed

echo "✓ Directories created: ocr_input/, ocr_output/, math_docs_processed/"

# 4. Verify math_doc_processor.py exists
if [ -f "math_doc_processor.py" ]; then
    echo "✓ math_doc_processor.py found"
else
    echo "⚠ math_doc_processor.py not found - please ensure it's in repo root"
fi

echo ""
echo "✅ Math RAG setup complete!"
echo ""
echo "📖 Next steps:"
echo "1. Copy your PDF files to: ocr_input/"
echo "2. Run OCR: ./process_math_docs.sh"
echo "3. Ingest into LightRAG via WebUI (http://localhost:8000)"
echo ""
echo "For detailed guide, see: MATH_RAG_GUIDE.md"
