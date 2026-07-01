#!/bin/bash
#SBATCH --job-name=reverify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/reverify_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/reverify_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

# Re-verify all "proved" results from miniF2F evaluations.
# Run on compute node for speed parity with original eval.

module load python/3.11.9

source /project/dachxiu/afeng/prover/venv/bin/activate

export PATH="$HOME/.elan/bin:$PATH"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

python3 /project/dachxiu/afeng/prover/scripts/reverify_results.py
