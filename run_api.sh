#!/usr/bin/env bash
# Khởi động API HealthRisk — chạy ngoài terminal của bạn thì sẽ không bị sập.
# Cách dùng:
#   ./run_api.sh start    # bật server (http://127.0.0.1:8000)
#   ./run_api.sh stop     # tắt server
#   ./run_api.sh restart  # bật lại
#   ./run_api.sh log      # xem log
set -e

PORT="${PORT:-8000}"
LOG=/tmp/opencode/api.log

start() {
    if curl -s -m 2 "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
        echo "Server đang chạy sẵn tại http://127.0.0.1:${PORT}"
        return 0
    fi
    cd "$(dirname "$0")"
    setsid nohup python3 -m uvicorn src.api:app --host 0.0.0.0 --port "${PORT}" > "${LOG}" 2>&1 < /dev/null &
    sleep 3
    if curl -s -m 5 "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
        echo "OK — Server chạy tại http://127.0.0.1:${PORT}"
        echo "Swagger: http://127.0.0.1:${PORT}/docs"
    else
        echo "Lỗi khởi động. Xem log: tail -50 ${LOG}"
        exit 1
    fi
}

stop() {
    pkill -f "uvicorn src.api" >/dev/null 2>&1 || true
    echo "Đã tắt server."
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 1; start ;;
    log)     tail -50 "${LOG}" ;;
    *)       echo "Dùng: ./run_api.sh {start|stop|restart|log}" ;;
esac
