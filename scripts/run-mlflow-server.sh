#!/bin/bash
# MLflow tracking 서버를 tmux 세션에서 백그라운드로 실행
# 저장소: config/.env에 저장소 위치 존재 → Backend Store: MLFLOW_BACKEND_STORE_URI(Neon) / Artifact Store: MLFLOW_ARTIFACTS_DESTINATION(S3)
# 사용법: bash scripts/run-mlflow-server.sh (중지: bash scripts/run-mlflow-server.sh stop)
set -euo pipefail
mkdir -p logs

##################### 스크립트 파일 위치 기준 프로젝트 루트로 이동 후 커맨드 실행 #####################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

##################### 설정 변수 #####################
HOST="127.0.0.1"      # 외부에 열지 않음 (Mac에서는 SSH 포트포워딩으로 접속)
PORT=5002             # config.yaml의 mlflow.tracking_uri와 같은 포트
SESSION="mlflow"      # tmux 세션 이름
ENV_FILE="config/.env"

##################### 로그 함수 #####################
log()  { echo -e "\n\033[1;32m[INFO]\033[0m $1"; }
warn() { echo -e "\n\033[1;33m[WARN]\033[0m $1"; }
die()  { echo -e "\n\033[1;31m[ERROR]\033[0m $1"; exit 1; }


##################### 중지 모드 #####################
if [ "${1:-start}" = "stop" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    pkill -f "mlflow.server.fastapi_app" 2>/dev/null || true   # tmux를 꺼도 살아남는 uvicorn 메인 + worker
    sleep 2
    if ss -ltn | grep -q ":$PORT "; then
        warn "$PORT 포트가 아직 사용 중입니다:"; ss -ltnp | grep ":$PORT "
        exit 1
    fi
    log "MLflow 서버 중지 완료."
    exit 0
fi


##################### 이미 떠 있으면 종료 #####################
if curl -sf "http://$HOST:$PORT/health" >/dev/null; then
    log "MLflow 서버가 이미 http://$HOST:$PORT 에서 실행 중입니다."
    exit 0
fi

##################### .env 불러오기 #####################
# MLflow 서버 프로세스가 S3에 접근하려면 AWS_*, MLFLOW_S3_ENDPOINT_URL이 환경변수로 있어야 함
[ -f "$ENV_FILE" ] || die "$ENV_FILE 파일이 없습니다."
set -a              # 이 사이에서 정의된 변수를 전부 export
source "$ENV_FILE"
set +a

[ -n "${MLFLOW_BACKEND_STORE_URI:-}" ]     || die "$ENV_FILE에 MLFLOW_BACKEND_STORE_URI가 비어 있습니다."
[ -n "${MLFLOW_ARTIFACTS_DESTINATION:-}" ] || die "$ENV_FILE에 MLFLOW_ARTIFACTS_DESTINATION이 비어 있습니다."

##################### tmux 세션에서 실행 #####################
if tmux has-session -t "$SESSION" 2>/dev/null; then
    warn "tmux 세션 '$SESSION'이 이미 있지만 서버 응답이 없습니다. 'tmux attach -t $SESSION'으로 확인하세요."
    exit 1
fi

log "MLflow 서버 시작 중... (tmux 세션: $SESSION)"
tmux new-session -d -s "$SESSION" \
    "uv run mlflow server \
        --host $HOST --port $PORT \
        --backend-store-uri '$MLFLOW_BACKEND_STORE_URI' \
        --artifacts-destination '$MLFLOW_ARTIFACTS_DESTINATION' \
     2>&1 | tee logs/mlflow-server.log"

##################### 뜰 때까지 대기 #####################
for _ in $(seq 1 30); do
    if curl -sf "http://$HOST:$PORT/health" >/dev/null; then
        log "MLflow 서버 실행 완료 → http://$HOST:$PORT"
        echo "  로그 보기: tmux attach -t $SESSION   (빠져나오기: Ctrl+b, d)"
        echo "  중지:      bash scripts/run-mlflow-server.sh stop"
        exit 0
    fi
    sleep 1
done
die "30초 안에 서버가 응답하지 않습니다. 'tmux attach -t $SESSION' 또는 logs/mlflow-server.log를 확인하세요."