#!/bin/bash
# Batch process math PDFs/Markdown with Nougat OCR or direct chunking

set -e

source .venv/bin/activate

INPUT_DIR="ocr_input"
OCR_OUTPUT_DIR="ocr_output"
PROCESSED_DIR="math_docs_processed"

# Check if input directory has files
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Directory not found: $INPUT_DIR"
    exit 1
fi

FILE_COUNT=$(find "$INPUT_DIR" -maxdepth 1 \( -name "*.pdf" -o -name "*.mmd" -o -name "*.md" -o -name "*.txt" \) | wc -l)
if [ $FILE_COUNT -eq 0 ]; then
    echo "❌ No files found in $INPUT_DIR/"
    echo "   Tip: Run 'python3 create_sample_math_docs.py' to create sample files"
    exit 1
fi

echo "📄 Processing $FILE_COUNT file(s)..."
echo ""

# Process each file
for input_file in "$INPUT_DIR"/*; do
    if [ -f "$input_file" ]; then
        filename=$(basename "$input_file")
        base_name="${filename%.*}"
        extension="${filename##*.}"
        
        echo "🔄 Processing: $filename"
        
        case "$extension" in
            pdf)
                # Try Nougat OCR (if available)
                if command -v nougat &> /dev/null || python3 -c "import nougat" 2>/dev/null; then
                    echo "   🔧 Running Nougat OCR..."
                    nougat "$input_file" -o "$OCR_OUTPUT_DIR" --pdf 2>/dev/null || {
                        echo "   ⚠️  Nougat OCR failed, trying ocrmypdf as fallback..."
                        if command -v ocrmypdf &> /dev/null; then
                            ocrmypdf -l vie+eng "$input_file" "$OCR_OUTPUT_DIR/${base_name}_ocr.pdf"
                            mmd_file="$OCR_OUTPUT_DIR/${base_name}_ocr.pdf"
                        else
                            echo "   ⚠️  No OCR tool available, skipping PDF"
                            continue
                        fi
                    }
                else
                    echo "   ⚠️  Nougat not installed, trying ocrmypdf..."
                    if command -v ocrmypdf &> /dev/null; then
                        ocrmypdf -l vie+eng "$input_file" "$OCR_OUTPUT_DIR/${base_name}_ocr.pdf"
                        mmd_file="$OCR_OUTPUT_DIR/${base_name}_ocr.pdf"
                    else
                        echo "   ❌ No OCR tool available (install: ocrmypdf or nougat)"
                        continue
                    fi
                fi
                
                # If ocrmypdf created file, extract to .txt for processing
                if [ -f "$mmd_file" ]; then
                    echo "   ✓ OCR completed, extracting text..."
                    pdftotext "$mmd_file" "$OCR_OUTPUT_DIR/${base_name}.txt"
                    mmd_file="$OCR_OUTPUT_DIR/${base_name}.txt"
                fi
                ;;
            
            mmd|md|txt)
                # Direct markdown/text processing (no OCR needed)
                mmd_file="$input_file"
                echo "   ✓ Processing markdown/text file directly"
                ;;
            
            *)
                echo "   ⚠️  Unsupported format: $extension (skipping)"
                continue
                ;;
        esac
        
        # Process with math_doc_processor
        if [ -f "$mmd_file" ]; then
            echo "   🔧 Chunking and indexing formulas..."
            python3 math_doc_processor.py "$mmd_file"
            
            # Move processed file to internal_docs for ingestion
            processed_file="$PROCESSED_DIR/${base_name}_processed.jsonl"
            if [ -f "$processed_file" ]; then
                cp "$processed_file" "internal_docs/"
                echo "   ✓ Copied to internal_docs/ for ingestion"
            fi
        fi
        
        echo ""
    fi
done

echo "✅ All files processed!"
echo ""
echo "📋 Summary:"
echo "   - Output: $PROCESSED_DIR/"
echo "   - Ready for ingestion: internal_docs/"
echo ""
echo "Next: Run 'python demo_math_rag.py' to ingest and query"
