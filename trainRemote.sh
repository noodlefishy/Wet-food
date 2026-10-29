#!/usr/bin/env bash
set -e

sessionName="pico-build-$(date +%s)"
IS_BENCHMARK=false

for arg in "$@"; do
    if [[ "$arg" == "--benchmark" ]] || [[ "$arg" == "--no-train" ]] || [[ "$arg" == "--eval-only" ]]; then
        IS_BENCHMARK=true
    fi
done

echo "==========================================================="
if [ "$IS_BENCHMARK" = true ]; then
    echo "   picoAI: REMOTE COLAB GPU BENCHMARK-ONLY RUNNER"
else
    echo "   picoAI: REMOTE COLAB GPU TRAIN & DOWNLOAD PIPELINE"
fi
echo "==========================================================="
echo "[*] Session: ${sessionName}"
echo "[*] Spinning up Google Colab GPU (T4)..."

.venv/bin/colab new --gpu T4 -s "${sessionName}"

cleanup() {
    echo ""
    echo "[*] Stopping remote Colab session..."
    .venv/bin/colab stop -s "${sessionName}" 2>/dev/null || true
    echo "[✓] Colab session stopped."
}
trap cleanup EXIT

echo "[*] Uploading runner script to Colab VM..."
.venv/bin/colab upload -s "${sessionName}" colabRunner.py /content/colabRunner.py

if [ "$IS_BENCHMARK" = true ]; then
    echo "[*] Setting benchmark flag on remote VM..."
    touch /tmp/benchmark_mode.flag
    .venv/bin/colab upload -s "${sessionName}" /tmp/benchmark_mode.flag /content/benchmark_mode.flag
fi

echo "[*] Executing pipeline on remote GPU..."
.venv/bin/colab exec -s "${sessionName}" -f colabRunner.py --timeout 1800

echo ""
echo "[*] Downloading results back to local disk..."
mkdir -p models/local firmware

# Download weights package if not in benchmark mode (or if updated)
.venv/bin/colab download -s "${sessionName}" /content/pico_weights_latest.zip models/local/pico_weights_latest.zip 2>/dev/null || true

# Download failures.json (now guaranteed to exist at /content/failures.json)
.venv/bin/colab download -s "${sessionName}" /content/failures.json failures.json 2>/dev/null || true

if [ -f "models/local/pico_weights_latest.zip" ]; then
    echo "[*] Unpacking weights to models/local and firmware/vocab.py..."
    unzip -o models/local/pico_weights_latest.zip -d models/local/ >/dev/null 2>&1
    cp models/local/vocab.py firmware/vocab.py 2>/dev/null || true
fi

echo ""
echo "==========================================================="
echo " [✓] REMOTE PIPELINE & LOCAL SYNC COMPLETE!"
if [ -f "failures.json" ]; then
    echo "     - failures.json synced ($(wc -l < failures.json) lines)"
fi
echo "==========================================================="