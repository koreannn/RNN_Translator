#!/bin/bash
set -euo pipefail

##################### 설정 변수 #####################
GIT_NAME="koreannn"
GIT_EMAIL="ghdtjdwo5@gmail.com"
PYTHON_VER="3.11"
USE_UV=true

##################### 로그 함수 #####################
log()  { echo -e "\n\033[1;32m[INFO]\033[0m $1"; }
warn() { echo -e "\n\033[1;33m[WARN]\033[0m $1"; }
die()  { echo -e "\n\033[1;31m[ERROR]\033[0m $1"; exit 1; }

##################### 시스템 패키지 #####################
log "시스템 패키지 설치 중..."
apt-get update -qq
apt-get install -y \
    sudo wget curl git vim \
    build-essential ca-certificates \
    tmux htop tree unzip

##################### uv 설치 #####################
if [ "$USE_UV" = true ]; then
    log "uv 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

    log "Python $PYTHON_VER 설치 중..."
    uv python install "$PYTHON_VER"
fi

##################### 의존성 설치 #####################
log "파이썬 패키지 설치 중..."
if [ "$USE_UV" = true ]; then
    if [ -f "uv.lock" ]; then
        log "uv.lock 감지 → uv sync 실행 중..."
        uv sync
    elif [ -f "pyproject.toml" ]; then
        log "pyproject.toml 감지 → uv lock 후 uv sync 실행 중..."
        uv lock
        uv sync
    elif [ -f "requirements.txt" ]; then
        warn "requirements.txt만 존재. uv add로 설치합니다."
        uv add -r requirements.txt
    else
        warn "의존성 파일 없음. 패키지 설치를 건너뜁니다."
    fi
fi

##################### .venv 자동 활성화 등록 #####################
# 스크립트 종료 후 새 터미널 진입 시 자동 활성화되도록 .bashrc에 등록
VENV_PATH="$(pwd)/.venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    log ".venv 자동 활성화 등록 중..."
    echo "source $VENV_PATH" >> ~/.bashrc
    log ".venv 자동 활성화 등록 완료 → 'source ~/.bashrc' 실행 후 적용됩니다."
else
    warn ".venv가 존재하지 않습니다. 활성화 등록을 건너뜁니다."
fi

##################### git 설정 #####################
log "git 전역 설정 중..."
git config --global user.name  "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
git config --global core.editor vim
git config --global init.defaultBranch main
git config --global credential.helper "cache --timeout=21600"

##################### 환경 확인 #####################
log "환경 확인 중..."
echo "----------------------------------------"
echo "Python   : $(python3 --version 2>/dev/null || echo '미설치')"
echo "CUDA     : $(nvcc --version 2>/dev/null | grep release || echo 'nvcc 없음')"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU 없음')"
if [ "$USE_UV" = true ]; then
    echo "uv       : $(uv --version)"
fi
echo "----------------------------------------"
log "세팅 완료. 아래 명령어를 실행하십시오."
echo ""
echo "    source ~/.bashrc"
echo ""