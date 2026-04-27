##################### 스크립트 파일 위치 기준 프로젝트 루트로 이동 후 커맨드 실행 #####################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"


##################### 로그 함수 #####################
log()  { echo -e "\n\033[1;32m[INFO]\033[0m $1"; }
warn() { echo -e "\n\033[1;33m[WARN]\033[0m $1"; }
die()  { echo -e "\n\033[1;31m[ERROR]\033[0m $1"; exit 1; }


##################### wandb 설정 #####################
WANDB_ENV_FILE=".env"

if [ -f "$WANDB_ENV_FILE" ]; then
    WANDB_API_KEY_VALUE=$(grep -E "^WANDB_API_KEY" "$WANDB_ENV_FILE" | cut -d '=' -f2 | tr -d ' ')
    if [ -n "$WANDB_API_KEY_VALUE" ]; then
        log "WandB 로그인 중..."
        wandb login --relogin "$WANDB_API_KEY_VALUE"
        log "WandB 로그인 완료."
    else
        warn ".env에 WANDB_API_KEY가 비어 있습니다. WandB 설정을 건너뜁니다."
    fi
else
    warn ".env 파일이 없습니다. WandB 설정을 건너뜁니다."
fi