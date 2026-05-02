#!/bin/bash

# ============================================================
# LightRAG Local Startup Script
# Khởi động tất cả services cần thiết
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
USE_DOCKER="true"

echo "╔════════════════════════════════════════════════════════╗"
echo "║        🚀 LightRAG Local Startup                       ║"
echo "║        Hoàn toàn miễn phí - Chạy local                 ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Function: Check command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ '$1' không tìm thấy. Vui lòng cài đặt:"
        echo "   $2"
        exit 1
    fi
}

# Function: Print section
print_section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ============================================================
# STEP 1: Check Prerequisites
# ============================================================
print_section "STEP 1: Kiểm tra yêu cầu"

check_command "python3" "sudo apt install python3 (Ubuntu/Debian)"
check_command "ollama" "curl -fsSL https://ollama.ai/install.sh | sh"

if command -v docker >/dev/null 2>&1 && command -v docker-compose >/dev/null 2>&1; then
    echo "✅ Docker version: $(docker --version)"
    echo "✅ Docker Compose available"
else
    USE_DOCKER="false"
    echo "⚠️  Docker không có sẵn, sẽ chạy chế độ local-only (không Neo4j/không container)."
fi

echo "✅ Python version: $(python3 --version)"
echo "✅ Ollama version: $(ollama --version 2>/dev/null || echo 'OK')"

# ============================================================
# STEP 2: Check Ollama Server
# ============================================================
print_section "STEP 2: Kiểm tra Ollama server"

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama server đang chạy"
    
    # Check models
    if curl -s http://localhost:11434/api/tags | grep -q "mistral"; then
        echo "✅ Model 'mistral' đã có"
    else
        echo "⚠️  Model 'mistral' chưa được pull"
        echo "   Chạy: ollama pull mistral"
    fi
    
    if curl -s http://localhost:11434/api/tags | grep -q "nomic-embed-text"; then
        echo "✅ Model 'nomic-embed-text' đã có"
    else
        echo "⚠️  Model 'nomic-embed-text' chưa được pull"
        echo "   Chạy: ollama pull nomic-embed-text"
    fi
else
    echo "❌ Ollama server không chạy!"
    echo ""
    echo "   ⚠️  Cần khởi động Ollama trong terminal khác:"
    echo "   $ ollama serve"
    echo ""
    echo "   Sau đó, pull models (nếu chưa có):"
    echo "   $ ollama pull mistral"
    echo "   $ ollama pull nomic-embed-text"
    echo ""
    read -p "Bạn đã khởi động Ollama? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "⚠️  Vui lòng khởi động Ollama trước!"
        exit 1
    fi
fi

# Keep the local configuration explicit even if the shell environment is empty.
export LLM_BINDING="${LLM_BINDING:-ollama}"
export LLM_BINDING_HOST="${LLM_BINDING_HOST:-http://localhost:11434}"
export LLM_MODEL="${LLM_MODEL:-mistral}"
export EMBEDDING_BINDING="${EMBEDDING_BINDING:-ollama}"
export EMBEDDING_BINDING_HOST="${EMBEDDING_BINDING_HOST:-http://localhost:11434}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
export LIGHTRAG_KV_STORAGE="${LIGHTRAG_KV_STORAGE:-JsonKVStorage}"
export LIGHTRAG_VECTOR_STORAGE="${LIGHTRAG_VECTOR_STORAGE:-NanoVectorDBStorage}"
export LIGHTRAG_GRAPH_STORAGE="${LIGHTRAG_GRAPH_STORAGE:-NetworkXStorage}"
export LIGHTRAG_DOC_STATUS_STORAGE="${LIGHTRAG_DOC_STATUS_STORAGE:-JsonDocStatusStorage}"
export WORKING_DIR="${WORKING_DIR:-./internal_docs}"

# ============================================================
# STEP 3: Setup Python Environment
# ============================================================
print_section "STEP 3: Cài đặt Python environment"

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Tạo virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment tạo xong"
else
    echo "✅ Virtual environment đã tồn tại"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"

# Install/upgrade pip
echo "📦 Cập nhật pip..."
python3 -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✅ pip cập nhật"

# ============================================================
# STEP 4: Install LightRAG
# ============================================================
print_section "STEP 4: Cài đặt LightRAG"

if [ -f "$PROJECT_DIR/pyproject.toml" ]; then
    echo "📦 Cài đặt LightRAG từ source..."
    cd "$PROJECT_DIR"
    pip install -e . > /dev/null 2>&1
    echo "✅ LightRAG cài đặt"
else
    echo "⚠️  pyproject.toml không tìm thấy!"
    exit 1
fi

# ============================================================
# STEP 5: Start Optional Docker Services
# ============================================================
print_section "STEP 5: Khởi động Docker services"

if [[ "$USE_DOCKER" == "true" ]]; then
    if [ ! -f "$PROJECT_DIR/docker-compose.local.yml" ]; then
        echo "❌ docker-compose.local.yml không tìm thấy!"
        exit 1
    fi

    echo "🐳 Khởi động Neo4j + Embedding model..."
    cd "$PROJECT_DIR"
    docker-compose -f docker-compose.local.yml down > /dev/null 2>&1 || true
    docker-compose -f docker-compose.local.yml up -d

    # Wait for services
    echo "⏳ Đợi services khởi động..."
    sleep 5

    # Check Neo4j
    if docker-compose -f docker-compose.local.yml exec neo4j cypher-shell -u neo4j -p lightrag_password_2024 "RETURN 1" > /dev/null 2>&1; then
        echo "✅ Neo4j sẵn sàng (port 7687)"
    else
        echo "⚠️  Neo4j chưa sẵn sàng, đợi thêm..."
        sleep 10
    fi

    echo "✅ Docker services đang chạy"
    echo "   - Neo4j Dashboard: http://localhost:7474"
    echo "   - Embedding API: http://localhost:8001"
else
    echo "ℹ️  Bỏ qua Docker services. LightRAG sẽ dùng backend mặc định local-only:"
    echo "   - KV: $LIGHTRAG_KV_STORAGE"
    echo "   - Vector: $LIGHTRAG_VECTOR_STORAGE"
    echo "   - Graph: $LIGHTRAG_GRAPH_STORAGE"
    echo "   - Doc Status: $LIGHTRAG_DOC_STATUS_STORAGE"
fi

# ============================================================
# STEP 6: Create Data Directory
# ============================================================
print_section "STEP 6: Chuẩn bị thư mục"

mkdir -p "$PROJECT_DIR/internal_docs"
echo "✅ Thư mục tài liệu: $PROJECT_DIR/internal_docs"

# ============================================================
# SUCCESS
# ============================================================
print_section "✅ SETUP HOÀN THÀNH"

echo ""
echo "🎯 Tiếp theo, chạy:"
echo ""
echo "   # Option 1: Test setup"
echo "   source $VENV_DIR/bin/activate"
echo "   python3 $PROJECT_DIR/demo_local.py"
echo ""
echo "   # Option 2: Chạy WebUI server"
echo "   source $VENV_DIR/bin/activate"
echo "   lightrag-server"
echo "   # Mở: http://localhost:8000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 Tài liệu:"
echo "   - Setup Guide: $PROJECT_DIR/SETUP_LOCAL_GUIDE.md"
echo "   - Neo4j Dashboard: http://localhost:7474"
echo "   - WebUI: http://localhost:8000 (sau khi chạy server)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
