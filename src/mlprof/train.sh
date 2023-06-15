#!/bin/bash --login
#
#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
PIDS=$(ps aux | grep -E "$USER.+mpi.+main.py" | grep -v grep | awk '{print $2}')
if [ -n "${PIDS}" ]; then
  echo "Already running! Exiting!"
  exit 1
fi

# DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

PARENT=$(dirname "${DIR}")
ROOT=$(dirname "${PARENT}")
echo "cwd: $DIR"
echo "parent: $PARENT"
echo "ROOT: $ROOT"
# printf '%.s─' $(seq 1 $(tput cols))

HOST=$(hostname)
TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: ${TSTAMP} on ${HOST}"

NCPU_PER_RANK=$(getconf _NPROCESSORS_ONLN)

if [[ $(hostname) == x* ]]; then
  # export LD_PRELOAD="/soft/perftools/mpitrace/lib/libmpitrace.so"
  ALCF_RESOURCE="polaris"
  HOSTFILE="${PBS_NODEFILE}"
  NRANKS=$(wc -l < "${PBS_NODEFILE}")
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS="$((NRANKS * NGPU_PER_RANK))"
  module load conda/2023-01-10-unstable ; conda activate base
  VENV_PREFIX="2023-01-10"
  MPI_COMMAND=$(which mpiexec)
  # --depth=${NCPU_PER_RANK} \
  MPI_FLAGS="--envall \
    -n ${NGPUS} \
    --ppn ${NGPU_PER_RANK} \
    --hostfile ${PBS_NODEFILE}"
elif [[ $(hostname) == theta* ]]; then
  ALCF_RESOURCE="thetaGPU"
  HOSTFILE="${COBALT_NODEFILE}"
  NRANKS=$(wc -l < "${COBALT_NODEFILE}")
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  module load conda/2023-01-11; conda activate base
  VENV_PREFIX="2023-01-11"
  MPI_COMMAND=$(which mpirun)
  MPI_FLAGS="-x LD_LIBRARY_PATH \
    -x PATH \
    -n ${NGPUS} \
    -npernode ${NGPU_PER_RANK} \
    --hostfile ${HOSTFILE}"
else
  ALCF_RESOURCE="NONE"
  HOSTFILE=/etc/hostname
  NRANKS=1
  NGPU_PER_RANK=0
  NGPUS=0
  echo "HOSTNAME: $(hostname)"
fi


export ALCF_RESOURCE="${ALCF_RESOURCE}"
export HOSTFILE="$HOSTFILE"
export NRANKS="${NRANKS}"
export NGPU_PER_RANK="${NGPU_PER_RANK}"
export NGPUS="${NGPUS}"
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┃  RUNNING ON ${ALCF_RESOURCE}: ${NGPUS} GPUs"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# -----------------------------------------------
# - Get number of global CPUs by multiplying: 
#        (# CPU / rank) * (# ranks)
# -----------------------------------------------
export NCPUS=$(("${NRANKS}"*"${NCPU_PER_RANK}"))


# ---- Specify directories and executable for experiment ------------------
MAIN="${DIR}/main.py"
LOGDIR="${DIR}/logs"
LOGFILE="${LOGDIR}/${TSTAMP}-${HOST}_ngpu${NGPUS}.log"
if [ ! -d "${LOGDIR}" ]; then
  mkdir -p "${LOGDIR}"
fi

# Keep track of latest logfile for easy access
echo "$LOGFILE" >> "${DIR}/logs/latest"

# Double check everythings in the right spot
echo "DIR=${DIR}"
echo "PARENT=${PARENT}"
echo "ROOT=${ROOT}"
echo "LOGDIR=${LOGDIR}"
echo "LOGFILE=${LOGFILE}"

# -----------------------------------------------------------
# 1. Check if a virtual environment exists in project root: 
#    `sdl_workshop/hyperparameterManagement/`
#
# 2. If so, activate environment and make sure we have an 
#    editable install
# -----------------------------------------------------------
VENV_DIR="${ROOT}/venvs/${ALCF_RESOURCE}/${VENV_PREFIX}"
if [ -d "${VENV_DIR}" ]; then
  echo "Found venv at: ${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
else
  echo "Creating new venv at: ${VENV_DIR}"
  python3 -m venv ${VENV_DIR} --system-site-packages
  source "${VENV_DIR}/bin/activate"
  python3 -m pip install --upgrade pip setuptools wheel
  python3 -m pip install -e "${ROOT}"
fi

# ---- Environment settings ------------------------------------------
# export NCCL_DEBUG=INFO
# export KMP_SETTINGS=TRUE
# export OMP_NUM_THREADS=16
# export TF_ENABLE_AUTO_MIXED_PRECISION=1

# ---- Define executable -----------------------------------
EXEC="${MPI_COMMAND} ${MPI_FLAGS} $(which python3) ${MAIN}"


# ------ Print job information --------------------------------------+
# printf '%.s─' $(seq 1 $(tput cols))
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┃  STARTING A NEW RUN ON ${NGPUS} GPUs of ${ALCF_RESOURCE}"
echo "┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┃  - Writing logs to: ${LOGFILE}"
echo "┃  - DATE: ${TSTAMP}"
echo "┃  - NRANKS: $NRANKS"
echo "┃  - NGPUS PER RANK: ${NGPU_PER_RANK}"
echo "┃  - NGPUS TOTAL: ${NGPUS}"
echo "┃  - python3: $(which python3)"
echo "┃  - MPI: ${MPI_COMMAND}"
echo "┃  - exec: ${EXEC} $@"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo 'To view output: `tail -f $(tail -1 logs/latest)`'
echo "Latest logfile: $(tail -1 ./logs/latest)"
echo "tail -f $(tail -1 logs/latest)"

${EXEC} "$@" > "${LOGFILE}" &
#LD_PRELOAD=/soft/perftools/mpitrace/lib/libmpitrace.so ${EXEC} $@ > ${LOGFILE}
