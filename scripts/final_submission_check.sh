#!/usr/bin/env bash
set -euo pipefail

echo "=============================="
echo "🚀 FINAL SUBMISSION CHECK START"
echo "=============================="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cleanup() {
  docker rm -f shopops-test >/dev/null 2>&1 || true
}
trap cleanup EXIT

# 1. Run tests
echo "🧪 Running tests..."
if [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
  (cd "${ROOT_DIR}" && "${PROJECT_ROOT}/venv/bin/python" -m pytest "${ROOT_DIR}/tests" -q)
else
  (cd "${ROOT_DIR}" && python -m pytest "${ROOT_DIR}/tests" -q)
fi

# 2. Check OpenEnv YAML exists
echo "📄 Checking openenv.yaml..."
if [ ! -f "${ROOT_DIR}/openenv.yaml" ]; then
  echo "❌ openenv.yaml missing"
  exit 1
fi

# 3. Docker build
echo "🐳 Building Docker..."
( cd "${ROOT_DIR}" && docker build -t shopops-env -f server/Dockerfile . )

echo "🐳 Running Docker container..."
docker run -d -p 8000:8000 --name shopops-test shopops-env

sleep 5

# 4. Health check
echo "💓 Checking health endpoint..."
HEALTH_BODY="$(mktemp)"
HEALTH_STATUS=""
for attempt in {1..10}; do
  HEALTH_STATUS="$(curl -s -o "${HEALTH_BODY}" -w "%{http_code}" http://localhost:8000/health || true)"
  if [ "${HEALTH_STATUS}" = "200" ]; then
    break
  fi
  sleep 1
done
if [ "${HEALTH_STATUS}" != "200" ]; then
  echo "❌ Health check failed (status=${HEALTH_STATUS})"
  echo "Health response:"
  cat "${HEALTH_BODY}"
  docker logs shopops-test
  exit 1
fi
echo "✅ Health check OK (${HEALTH_STATUS})"

# 5. Reset endpoint
echo "🔄 Testing reset()..."
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"seed":42}' | grep -i "observation" || {
  echo "❌ reset() failed"
  docker logs shopops-test
  exit 1
}

# 6. Inference check
echo "🤖 Running inference..."

export API_BASE_URL=${API_BASE_URL:-"https://api.openai.com/v1"}
export MODEL_NAME=${MODEL_NAME:-"gpt-4o"}
export HF_TOKEN=${HF_TOKEN:-"test_key"}
export ENV_URL=${ENV_URL:-"http://localhost:8000"}
export SEED=${SEED:-42}

if [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
  "${PROJECT_ROOT}/venv/bin/python" "${ROOT_DIR}/inference.py" > "${ROOT_DIR}/outputs/inference_output.txt"
else
  python "${ROOT_DIR}/inference.py" > "${ROOT_DIR}/outputs/inference_output.txt"
fi

# 7. Validate strict logs
echo "📊 Validating log format..."

grep "\[START\]" "${ROOT_DIR}/outputs/inference_output.txt" > /dev/null || { echo "❌ Missing [START]"; exit 1; }
grep "\[STEP\]" "${ROOT_DIR}/outputs/inference_output.txt" > /dev/null || { echo "❌ Missing [STEP]"; exit 1; }
grep "\[END\]" "${ROOT_DIR}/outputs/inference_output.txt" > /dev/null || { echo "❌ Missing [END]"; exit 1; }

echo "📈 Sample Output:"
head -20 "${ROOT_DIR}/outputs/inference_output.txt"

echo "=============================="
echo "✅ ALL CHECKS PASSED"
echo "🚀 READY FOR SUBMISSION"
echo "============"
