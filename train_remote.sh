#!/usr/bin/env bash
set -e

SESSION_NAME="pico-build-$(date +%s)"

echo "==========================================================="
echo "   picoAI: REMOTE COLAB GPU TRAIN & DOWNLOAD PIPELINE"
echo "==========================================================="
echo "[*] Session: ${SESSION_NAME}"
echo "[*] Spinning up Google Colab GPU (T4)..."

# 1. Create a remote session with T4 GPU
.venv/bin/colab new --gpu T4 -s "${SESSION_NAME}"

# Cleanup trap: ensure session is always stopped even on error or Ctrl+C
cleanup() {
    echo ""
    echo "[*] Stopping remote Colab session..."
    .venv/bin/colab stop -s "${SESSION_NAME}" 2>/dev/null || true
    echo "[✓] Colab session stopped."
}
trap cleanup EXIT

# 2. Upload runner script to the remote VM
echo "[*] Uploading runner script to Colab VM..."
.venv/bin/colab upload -s "${SESSION_NAME}" colab_runner.py /content/colab_runner.py

# 3. Execute colab_runner.py on the remote GPU (1800s timeout = 30 mins)
echo "[*] Executing build and test pipeline on remote GPU..."
.venv/bin/colab exec -s "${SESSION_NAME}" -f colab_runner.py --timeout 1800

# 4. Download the trained weights package to local machine
echo ""
echo "[*] Downloading trained weights and vocab back to local disk..."
mkdir -p models/local firmware
.venv/bin/colab download -s "${SESSION_NAME}" /content/pico_weights_latest.zip models/local/pico_weights_latest.zip

# 4. Extract weights into place
echo "[*] Unpacking weights to models/local and firmware/vocab.py..."
unzip -o models/local/pico_weights_latest.zip -d models/local/
cp models/local/vocab.py firmware/vocab.py

echo ""
echo "==========================================================="
echo " [✓] REMOTE BUILD & LOCAL SYNC COMPLETE!"
echo "     - models/local/pico1 weights.model"
echo "     - models/local/pico2 weights.model"
echo "     - models/local/vocab.py & firmware/vocab.py"
echo "==========================================================="
