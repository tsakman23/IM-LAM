#!/usr/bin/bash
# =================================================
# Submit a command to SLURM on HoreKa (NHR@KIT)
# =================================================
#
# Cluster defaults live in the variables below. Optional flags override them
# for a single invocation. No container / Pyxis — runs in your environment
# from the directory where you invoke sbatch (see --wrap cd).
#
# Usage (prepend to your training command):
#   ./scripts/horeka.sh python experiments/run_lapo_bc.py run_id=foo ...
#
# With one-off overrides:
#   ./scripts/horeka.sh --job_name=lapo_q1 --time=1-00:00:00 -- python experiments/run_lapo_bc.py ...
#
# Docs: https://www.nhr.kit.edu/userdocs/horeka/batch/
#

# ---------------------------------------------------------------------------
# HoreKa defaults — edit for your project / queue
# ---------------------------------------------------------------------------
HOREKA_PARTITION="accelerated"
HOREKA_JOB_NAME="imitation"
# Wall time (HoreKa GPU queues allow up to 2-00:00:00)
HOREKA_TIME="24:00:00"
# CPUs per task (accelerated default: 38 per GPU when gres=gpu:1)
HOREKA_CPUS_PER_TASK=38
HOREKA_GPUS=1
# Memory in MB for --mem=... Leave empty to omit (Slurm uses partition defaults; recommended on HoreKa).
HOREKA_MEM_MB="124375"
# Slurm account; required if your user is in multiple projects. Leave empty to omit -A.
HOREKA_ACCOUNT=""
# Optional constraint, e.g. LSDF for LSDF-capable nodes. Leave empty to omit.
HOREKA_CONSTRAINT=""
# Optional dependency, e.g. afterok:12345
HOREKA_DEPENDENCY=""
# Log files (Slurm expands %j). Set empty strings to use sbatch defaults.
HOREKA_OUTPUT=""
HOREKA_ERROR=""

# Internal state (initialized from defaults; CLI may override)
PARTITION="$HOREKA_PARTITION"
JOB_NAME="$HOREKA_JOB_NAME"
TIME="$HOREKA_TIME"
CPUS="$HOREKA_CPUS_PER_TASK"
MEM="$HOREKA_MEM_MB"
GPUS="$HOREKA_GPUS"
ACCOUNT="$HOREKA_ACCOUNT"
CONSTRAINT="$HOREKA_CONSTRAINT"
DEPENDENCY="$HOREKA_DEPENDENCY"
OUTPUT="$HOREKA_OUTPUT"
ERROR="$HOREKA_ERROR"

show_help() {
    echo "Usage: $0 [OPTIONS] -- COMMAND [ARGS...]"
    echo
    echo "Options (override ${0##*/} defaults at the top of this script):"
    echo "  --partition NAME     Slurm partition (default: \$HOREKA_PARTITION)"
    echo "  --job_name NAME      Job name (default: \$HOREKA_JOB_NAME)"
    echo "  --time LIMIT         Wall time, e.g. 12:00:00 or 1-00:00:00"
    echo "  --cpus N             --cpus-per-task (default: \$HOREKA_CPUS_PER_TASK)"
    echo "  --mem MB             Total job memory in MB; omit flag to use partition default"
    echo "  --gpus N             --gres=gpu:N (default: \$HOREKA_GPUS)"
    echo "  --account PROJECT    Slurm -A / --account (if needed)"
    echo "  --constraint SPEC    e.g. LSDF"
    echo "  --dependency SPEC    e.g. afterok:12345"
    echo "  --output PATTERN     sbatch --output (default: JOB_NAME_%j.out)"
    echo "  --error PATTERN      sbatch --error (default: JOB_NAME_%j.err)"
    echo "  -h, --help           Show this help"
    echo
    echo "Use -- before the command if any argument starts with -."
    exit 0
}

# Parse arguments; stop at first non-option or after --
REMAINING=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --partition=*)
            PARTITION="${1#*=}"
            shift
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --job_name=*)
            JOB_NAME="${1#*=}"
            shift
            ;;
        --job_name)
            JOB_NAME="$2"
            shift 2
            ;;
        --time=*)
            TIME="${1#*=}"
            shift
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --cpus=*)
            CPUS="${1#*=}"
            shift
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --mem=*)
            MEM="${1#*=}"
            shift
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#*=}"
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --account=*)
            ACCOUNT="${1#*=}"
            shift
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --constraint=*)
            CONSTRAINT="${1#*=}"
            shift
            ;;
        --constraint)
            CONSTRAINT="$2"
            shift 2
            ;;
        --dependency=*)
            DEPENDENCY="${1#*=}"
            shift
            ;;
        --dependency)
            DEPENDENCY="$2"
            shift 2
            ;;
        --output=*)
            OUTPUT="${1#*=}"
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --error=*)
            ERROR="${1#*=}"
            shift
            ;;
        --error)
            ERROR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        --)
            shift
            REMAINING+=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Try: $0 --help" >&2
            exit 1
            ;;
        *)
            REMAINING+=("$@")
            break
            ;;
    esac
done

set -- "${REMAINING[@]}"

if [[ "$#" -eq 0 ]]; then
    echo "Error: missing COMMAND. Example: $0 python experiments/run_lapo_bc.py ..." >&2
    echo "Try: $0 --help" >&2
    exit 1
fi

if [[ -z "$PARTITION" || -z "$JOB_NAME" ]]; then
    echo "Error: partition and job_name must be non-empty (set HOREKA_* at top of script or pass flags)." >&2
    exit 1
fi

SUBMIT_DIR=$(pwd)
quoted_cmd=()
for arg in "$@"; do
    quoted_cmd+=("$(printf '%q' "$arg")")
done
cmd_line="${quoted_cmd[*]}"

WRAP="set -euo pipefail; cd $(printf '%q' "$SUBMIT_DIR"); $cmd_line"

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="${JOB_NAME}_%j.out"
fi
if [[ -z "$ERROR" ]]; then
    ERROR="${JOB_NAME}_%j.err"
fi

SBATCH_CMD=(
    sbatch
    -p "${PARTITION}"
    --job-name="${JOB_NAME}"
    --output="${OUTPUT}"
    --error="${ERROR}"
    --time="${TIME}"
    --cpus-per-task="${CPUS}"
    --gres="gpu:${GPUS}"
    --ntasks=1
    --wrap="${WRAP}"
)

if [[ -n "$MEM" ]]; then
    SBATCH_CMD+=(--mem="${MEM}")
fi

if [[ -n "$ACCOUNT" ]]; then
    SBATCH_CMD+=(--account="${ACCOUNT}")
fi

if [[ -n "$CONSTRAINT" ]]; then
    SBATCH_CMD+=(--constraint="${CONSTRAINT}")
fi

if [[ -n "$DEPENDENCY" ]]; then
    SBATCH_CMD+=(--dependency="${DEPENDENCY}")
fi

"${SBATCH_CMD[@]}"
