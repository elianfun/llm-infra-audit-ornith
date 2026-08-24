#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# vLLM 雙機叢集守護 —— 開機自啟 + 定期健康檢查 + 優雅重啟
#
#   vllm-guard.sh boot    @reboot 用:先等 worker 起來再啟動
#   vllm-guard.sh check   cron 每 2 分鐘:連續 N 次生成失敗才重啟
#
#   維護前先 touch ~/.vllm-guard-off  (做完記得刪掉!)
#   設定值在同目錄的 vllm-guard.conf
#
# 為什麼不直接 docker stop:容器 PID 1 是 sleep infinity,vLLM 是 docker exec
# 進去跑的 → docker stop 的 SIGTERM 只打到 sleep,vLLM 被硬砍,CUDA 沒收乾淨
# 會讓幾十 G 卡在驅動保留區,只能重開機。所以一定要先對 vllm serve 送 SIGTERM。
# ──────────────────────────────────────────────────────────────────────
#
# [此檔案為稽核時取得的複本，來源: /home/vghks7/vllm-guard.sh (172.17.161.28)]
# 依賴同目錄的 vllm-guard.conf（未備份，內含 API_KEY/MODEL/WORKER_IP/RECIPE/NODES 等變數，
# 請依下方使用到的變數名稱自行在新環境建立）
#
set -u

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SELF_DIR/vllm-guard.conf"

LOG=$HOME/vllm-guard.log
FAILS_F=$HOME/.vllm-guard-fails
OFF=$HOME/.vllm-guard-off
LOCK=$HOME/.vllm-guard.lock
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no"

: "${CHECK_TIMEOUT:=120}"
: "${FAIL_THRESHOLD:=4}"
: "${READY_TRIES:=60}"
: "${EXTRA_ARGS:=}"
: "${API_KEY:=}"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 維護旗標。也認舊看門狗的檔名 —— .30 原本用 .ornith-watchdog-off,
# 萬一有人照舊習慣 touch 那個檔,不能讓這支還跑去搶。
[ -f "$OFF" ] && exit 0
[ -f "$HOME/.ornith-watchdog-off" ] && exit 0

# boot 與 check 共用同一把鎖 → 開機時兩邊同時觸發也不會重複啟動
exec 9>"$LOCK"
flock -n 9 || exit 0

PAYLOAD="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}"

gen_check() {
  if [ -n "$API_KEY" ]; then
    curl -s -o /dev/null -w "%{http_code}" -m "$CHECK_TIMEOUT" \
      -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
      -d "$PAYLOAD" "http://127.0.0.1:$PORT/v1/chat/completions"
  else
    curl -s -o /dev/null -w "%{http_code}" -m "$CHECK_TIMEOUT" \
      -H "Content-Type: application/json" \
      -d "$PAYLOAD" "http://127.0.0.1:$PORT/v1/chat/completions"
  fi
}

mem_used() {   # $1 空=本機,否則 worker IP
  if [ -z "$1" ]; then free -g | awk '/^Mem:/{print $3}'
  else $SSH "$1" free -g 2>/dev/null | awk '/^Mem:/{print $3}'; fi
}

# 對某個節點容器內的 vllm serve 送 SIGTERM
# ray 模式的 worker 沒有 vllm serve(GPU 工作是 ray actor,由 head 收掉)→ 找不到是正常的
sigterm_node() {
  local tag=${1:-head} runner="" pid=""
  [ -n "${2:-}" ] && runner="$SSH $2"
  pid=$($runner docker exec vllm_node ps -eo pid,args --no-headers 2>/dev/null \
        | grep -F "vllm serve" | awk '{print $1}' | head -1)
  if [ -z "$pid" ]; then
    log "  $tag:沒有 vllm serve 行程(ray worker 本來就沒有,或已經死了)"
    return
  fi
  log "  $tag:SIGTERM vllm serve pid=$pid"
  $runner docker exec vllm_node kill -TERM "$pid" >/dev/null 2>&1
}

graceful_stop() {
  log "優雅停止中(先 SIGTERM,再 docker stop)"
  sigterm_node head ""
  sigterm_node worker "$WORKER_IP"

  local i h
  for i in $(seq 1 18); do
    sleep 5
    h=$(mem_used "")
    [ -n "$h" ] && [ "$h" -lt 15 ] && break
  done

  h=$(mem_used "")
  local w; w=$(mem_used "$WORKER_IP")
  log "  回收後記憶體 head=${h:-?}G worker=${w:-?}G(正常應該 <10G)"
  if [ -n "$w" ] && [ "$w" -gt 25 ] 2>/dev/null; then
    log "  ⚠️ worker 記憶體沒回收 —— 驅動洩漏,重啟很可能失敗,需要人工重開機"
  fi

  docker stop vllm_node >/dev/null 2>&1
  docker rm -f vllm_node >/dev/null 2>&1
  $SSH "$WORKER_IP" docker stop vllm_node >/dev/null 2>&1
  $SSH "$WORKER_IP" docker rm -f vllm_node >/dev/null 2>&1
}

save_evidence() {
  local d=$HOME/vllm-crash-logs/$(date '+%F_%H%M%S')
  mkdir -p "$d"
  docker logs --tail 400 vllm_node > "$d/head-container.log" 2>&1
  $SSH "$WORKER_IP" docker logs --tail 400 vllm_node > "$d/worker-container.log" 2>&1
  journalctl -k --since "2 hours ago" --no-pager 2>/dev/null \
    | grep -iE "nvrm|oom|mlx5|xid" | tail -50 > "$d/local-kernel.log"
  free -g > "$d/mem-head.txt" 2>&1
  $SSH "$WORKER_IP" free -g > "$d/mem-worker.txt" 2>&1
  log "  證據存到 $d"
}

relaunch() {
  cd "$HOME/spark-vllm-docker" || { log "❌ 找不到 $HOME/spark-vllm-docker"; return 1; }
  export VLLM_SPARK_EXTRA_DOCKER_ARGS="-v $MOUNT:/models"
  log "啟動:./run-recipe.sh $RECIPE $RAY_ARGS -n $NODES $EXTRA_ARGS -d"
  ./run-recipe.sh "$RECIPE" $RAY_ARGS -n "$NODES" $EXTRA_ARGS -d >> "$LOG" 2>&1
}

wait_ready() {
  local i code
  for i in $(seq 1 "$READY_TRIES"); do
    code=$(gen_check)
    if [ "$code" = "200" ]; then
      log "✅ 服務已就緒(等了約 $((i * 10)) 秒)"
      return 0
    fi
    sleep 10
  done
  log "❌ 等了 $((READY_TRIES * 10)) 秒還沒就緒(最後 http=$code)"
  return 1
}

case "${1:-check}" in
  selftest)
    echo "===== vllm-guard selftest($MODEL)====="
    echo "[1] 設定讀取"
    echo "    MODEL=$MODEL PORT=$PORT WORKER_IP=$WORKER_IP"
    echo "    RAY_ARGS=$RAY_ARGS  EXTRA_ARGS=[$EXTRA_ARGS]"
    echo "    FAIL_THRESHOLD=$FAIL_THRESHOLD  READY_TRIES=$READY_TRIES(最長等 $((READY_TRIES * 10)) 秒)"
    echo "[2] 健康檢查(check 與 wait_ready 都靠這個)"
    c=$(gen_check); echo "    gen_check http=$c  $([ "$c" = 200 ] && echo ✅ || echo ❌)"
    echo "[3] worker 連通性(重啟前會先擋這關)"
    ping -c1 -W3 "$WORKER_IP" >/dev/null 2>&1 && echo "    ping ✅" || echo "    ping ❌"
    hn=$($SSH "$WORKER_IP" hostname 2>&1 | tail -1); echo "    ssh → $hn"
    echo "[4] SIGTERM 目標偵測(只查 pid,不送訊號)"
    for pair in "head:" "worker:$WORKER_IP"; do
      tag=${pair%%:*}; ip=${pair#*:}; runner=""
      [ -n "$ip" ] && runner="$SSH $ip"
      pid=$($runner docker exec vllm_node ps -eo pid,args --no-headers 2>/dev/null \
            | grep -F "vllm serve" | awk '{print $1}' | head -1)
      if [ -n "$pid" ]; then echo "    $tag: 找到 vllm serve pid=$pid ✅"
      else echo "    $tag: 沒有 vllm serve(ray 模式的 worker 正常如此)"; fi
    done
    echo "[5] 重啟素材"
    echo "    recipe   $([ -f "$HOME/spark-vllm-docker/$RECIPE" ] && echo ✅ || echo ❌)  $RECIPE"
    echo "    模型目錄 $([ -d "$MOUNT" ] && echo ✅ || echo ❌)  $MOUNT"
    echo "    run-recipe.sh $([ -x "$HOME/spark-vllm-docker/run-recipe.sh" ] && echo ✅ || echo ❌)"
    echo "    worker 也看得到模型 $($SSH "$WORKER_IP" test -d "$MOUNT" && echo ✅ || echo ❌)"
    echo "    會執行:./run-recipe.sh $RECIPE $RAY_ARGS -n $NODES $EXTRA_ARGS -d"
    echo "[6] 存證功能(真的寫一份到 selftest 目錄)"
    d=$HOME/vllm-crash-logs/selftest
    rm -rf "$d"; mkdir -p "$d"
    docker logs --tail 20 vllm_node > "$d/head-container.log" 2>&1
    $SSH "$WORKER_IP" docker logs --tail 20 vllm_node > "$d/worker-container.log" 2>&1
    free -g > "$d/mem-head.txt" 2>&1
    $SSH "$WORKER_IP" free -g > "$d/mem-worker.txt" 2>&1
    for f in head-container.log worker-container.log mem-head.txt mem-worker.txt; do
      sz=$(stat -c %s "$d/$f" 2>/dev/null || echo 0)
      echo "    $f = ${sz} bytes $([ "$sz" -gt 0 ] && echo ✅ || echo ❌空的)"
    done
    echo "[7] 鎖與旗標"
    echo "    flock $([ -x /usr/bin/flock ] && echo ✅ || echo ❌)"
    echo "    維護旗標目前 $([ -f "$OFF" ] && echo '存在 → guard 是停用的!' || echo '不存在 → guard 啟用中 ✅')"
    echo "    crontab guard 行數 = $(crontab -l 2>/dev/null | grep -c vllm-guard)"
    echo "===== selftest 結束(沒有停任何服務)====="
    ;;

  boot)
    log "===== 開機啟動($MODEL)====="
    if [ "$(gen_check)" = "200" ]; then
      log "服務已經在跑,不用啟動"; exit 0
    fi
    log "等 worker $WORKER_IP 上線(ConnectX 可能比主機慢)"
    for i in $(seq 1 60); do
      $SSH "$WORKER_IP" hostname >/dev/null 2>&1 && break
      sleep 10
    done
    if ! $SSH "$WORKER_IP" hostname >/dev/null 2>&1; then
      log "❌ worker 10 分鐘內連不上,放棄(看門狗每 2 分鐘會再試)"
      exit 1
    fi
    log "worker 已上線"
    relaunch && wait_ready
    ;;

  check)
    code=$(gen_check)
    if [ "$code" = "200" ]; then
      rm -f "$FAILS_F"
      exit 0
    fi
    fails=$(( $(cat "$FAILS_F" 2>/dev/null || echo 0) + 1 ))
    echo "$fails" > "$FAILS_F"
    log "生成檢查失敗 http=$code ($fails/$FAIL_THRESHOLD)"
    [ "$fails" -lt "$FAIL_THRESHOLD" ] && exit 0
    rm -f "$FAILS_F"

    log "===== 連續 $FAIL_THRESHOLD 次失敗 → 完整重啟($MODEL)====="
    if ! ping -c1 -W3 "$WORKER_IP" >/dev/null 2>&1; then
      log "worker $WORKER_IP ping 不到 —— 先不動(半殘狀態亂重啟只會更糟)"
      exit 1
    fi
    save_evidence
    graceful_stop
    relaunch && wait_ready
    ;;

  *)
    echo "用法: $0 {boot|check}" >&2; exit 2 ;;
esac
