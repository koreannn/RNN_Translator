#!/bin/bash
# 기본 환경 세팅(setup-gpu-server.sh) 이후 실행하는 유틸성 세팅
# 사용 전: config/.env에 필요한 값(HF_TOKEN 등)을 직접 입력
set -uo pipefail

##################### 스크립트 파일 위치 기준 프로젝트 루트로 이동 후 커맨드 실행 #####################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"


##################### 로그 함수 #####################
log()  { echo -e "\n\033[1;32m[INFO]\033[0m $1"; }
warn() { echo -e "\n\033[1;33m[WARN]\033[0m $1"; }
die()  { echo -e "\n\033[1;31m[ERROR]\033[0m $1"; exit 1; }


##################### .env 값 읽기 #####################
ENV_FILE="config/.env"

# 사용법: read_env <키> → 값 출력 (앞뒤 공백·따옴표 제거). .env 전체를 source하지 않고 필요한 키만 읽음
read_env() {
    grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d '=' -f2- | sed -E 's/^[[:space:]]+|[[:space:]]+$//g; s/^"(.*)"$/\1/; s/^'"'"'(.*)'"'"'$/\1/'
}

[ -f "$ENV_FILE" ] || die "$ENV_FILE 파일이 없습니다. setup-gpu-server.sh를 먼저 실행하세요."


##################### Hugging Face 로그인 (비공개 데이터셋 접근용) #####################
# 로그인하면 토큰이 ~/.cache/huggingface/token에 저장되고, load_dataset·push_to_hub가 자동으로 사용
HF_TOKEN_VALUE="$(read_env HF_TOKEN)"

if [ -n "$HF_TOKEN_VALUE" ]; then
    log "Hugging Face 로그인 중..."
    if HF_LOGIN_ERR="$(uv run hf auth login --token "$HF_TOKEN_VALUE" 2>&1 >/dev/null)"; then
        log "Hugging Face 로그인 완료. (계정: $(uv run hf auth whoami 2>/dev/null | head -1))"
    else
        # 실패 시 traceback 전체 대신 마지막 오류 한 줄만 표시 (토큰 값은 출력하지 않음)
        warn "Hugging Face 로그인 실패: $(echo "$HF_LOGIN_ERR" | tail -1)\n       $ENV_FILE의 HF_TOKEN 값을 확인하세요."
    fi
else
    warn "$ENV_FILE에 HF_TOKEN이 비어 있습니다. Hugging Face 로그인을 건너뜁니다."
fi
