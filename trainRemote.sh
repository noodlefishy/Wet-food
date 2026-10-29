#!/usr/bin/env bash
set -e

sessionName="pico-build-$(date +%s)"

echo "==========================================================="
echo "   picoAI: REMOTE COLAB GPU TRAIN & DOWNLOAD PIPELINE"
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

echo "[*] Executing build and test pipeline on remote GPU..."
.venv/bin/colab exec -s "${sessionName}" -f colabRunner.py --timeout 1800

echo ""
echo "[*] Downloading trained weights and vocab back to local disk..."
mkdir -p models/local firmware
.venv/bin/colab download -s "${sessionName}" /content/pico_weights_latest.zip models/local/pico_weights_latest.zip
.venv/bin/colab download -s "${sessionName}" /content/failures.json failures.json

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
