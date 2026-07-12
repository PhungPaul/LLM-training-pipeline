#!/usr/bin/env bash
#
# setup_ec2.sh — provisions a g5.xlarge for the Supernova (GPT-500M) training run
#                and launches training in a detachable tmux session.
#
# Usage (run this ON the EC2 instance, after SSHing in):
#   scp -i your-key.pem philiane_supernova_505m_training_ec2.ipynb setup_ec2.sh ec2-user@<instance-ip>:~/
#   ssh -i your-key.pem ec2-user@<instance-ip>
#   chmod +x setup_ec2.sh
#   ./setup_ec2.sh
#
# ── Before you run this ────────────────────────────────────────────────────
#  1. Launch a g5.xlarge with the "Deep Learning AMI GPU PyTorch ... (Ubuntu/AL2)"
#     — this ships CUDA/cuDNN/driver already installed and version-matched,
#     which is the single biggest source of pain if you install manually.
#  2. Request a G/VT instance quota increase FIRST if this is a new AWS
#     account — default is 0 vCPUs for GPU families. Service Quotas console
#     -> "Running On-Demand G and VT instances". Can take hours to approve.
#  3. Create an S3 bucket for checkpoints/stats, and attach an IAM role to
#     the instance (not a static access key) with s3:GetObject/PutObject/
#     ListBucket/DeleteObject/CopyObject scoped to that bucket. boto3 in the
#     notebook picks up instance-role credentials automatically — no keys
#     to manage or leak.
#  4. Edit S3_BUCKET / S3_PREFIX at the top of the notebook's config cell
#     before running.
set -euo pipefail

echo "════════════════════════════════════════════════════════════"
echo " Supernova (GPT-500M) — EC2 setup"
echo "════════════════════════════════════════════════════════════"

# ── 1. Sanity check: GPU visible? ─────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Did you launch a GPU instance with a"
  echo "       Deep Learning AMI? Aborting."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── 2. Python deps ─────────────────────────────────────────────────────────
# Deep Learning AMIs usually ship a conda env with torch preinstalled
# (commonly `pytorch` or `pytorch_p310`) — activate it if present so we
# don't fight the AMI's own CUDA-matched torch build.
if command -v conda &>/dev/null; then
  ENV_NAME=$(conda env list | awk '/pytorch/{print $1; exit}')
  if [ -n "${ENV_NAME:-}" ]; then
    echo "Activating conda env: $ENV_NAME"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
  fi
fi

pip install -q --upgrade pip
pip install -q datasets transformers tokenizers bitsandbytes pynvml boto3 \
               jupyter nbconvert ipykernel tiktoken

# ── 3. Working directories (match the notebook's CKPT_DIR/CACHE_DIR) ──────
mkdir -p /home/ec2-user/gpt500m/checkpoints
mkdir -p /home/ec2-user/gpt500m/hf_cache

# ── 4. Verify S3 access via the instance's IAM role ────────────────────────
echo "Verifying S3 access (instance role credentials)..."
python3 - <<'PYEOF'
import boto3, sys
try:
    sts = boto3.client("sts")
    ident = sts.get_caller_identity()
    print(f"  IAM identity: {ident['Arn']}")
except Exception as e:
    print(f"  WARNING: couldn't resolve instance credentials: {e}")
    print("  Did you attach an IAM role with S3 permissions to this instance?")
    sys.exit(1)
PYEOF

# ── 5. Convert notebook to a plain script for the detachable run ──────────
NB_PATH="./philiane_supernova_505m_training_ec2.ipynb"
if [ ! -f "$NB_PATH" ]; then
  echo "ERROR: $NB_PATH not found in current directory. scp it here first."
  exit 1
fi

echo "Converting notebook -> script (jupyter over SSH is fine for interactive"
echo "work, but a long unattended run should live in tmux, not a kernel tied"
echo "to your SSH session)..."
jupyter nbconvert --to script "$NB_PATH" --output train_supernova

echo ""
echo "════════════════════════════════════════════════════════════"
echo " Setup complete."
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Edit S3_BUCKET / S3_PREFIX in train_supernova.py if you haven't"
echo "     already edited them in the notebook before converting."
echo "  2. Start a detachable training session:"
echo "       tmux new -s train"
echo "       python3 train_supernova.py"
echo "     Detach with Ctrl+B then D. Training keeps running after you"
echo "     close your laptop or lose SSH."
echo "  3. Reattach anytime with:"
echo "       tmux attach -t train"
echo "  4. Monitor progress without attaching:"
echo "       aws s3 cp s3://<bucket>/<prefix>/training_stats.csv - | tail -20"
