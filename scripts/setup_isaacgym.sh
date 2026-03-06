# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Use CONDA_ENV_NAME if provided, otherwise default to "hsgym"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsgym}
echo "conda environment name is set to: $CONDA_ENV_NAME"
# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_$CONDA_ENV_NAME
CONDA_SENTINEL=${WORKSPACE_DIR}/.env_setup_conda_$CONDA_ENV_NAME
ISAACGYM_SENTINEL=${WORKSPACE_DIR}/.env_setup_isaacgym_pkg_$CONDA_ENV_NAME
HOLOSOMA_SENTINEL=${WORKSPACE_DIR}/.env_setup_holosoma_pkg_$CONDA_ENV_NAME

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  if [[ ! -f $CONDA_SENTINEL ]]; then
    # Create the conda environment
    if [[ ! -d $ENV_ROOT ]]; then
      $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
      $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
      $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
      MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n $CONDA_ENV_NAME python=3.8 -c conda-forge --override-channels
    fi
    touch $CONDA_SENTINEL
  fi

  source $CONDA_ROOT/bin/activate $CONDA_ENV_NAME

  if [[ ! -f $ISAACGYM_SENTINEL ]]; then
    # Preinstall evdev from conda-forge to avoid pip source-build failures with conda toolchain.
    conda install -c conda-forge -y evdev

    # Best-effort fallback for systems that still need crypt headers.
    if command -v apt-get &> /dev/null; then
      sudo apt-get update || true
      sudo apt-get install -y libxcrypt-dev || sudo apt-get install -y libcrypt-dev || true
    fi

    # Install libstdcxx-ng to fix the error: `version `GLIBCXX_3.4.32' not found` on Ubuntu 24.04
    conda install -c conda-forge -y libstdcxx-ng

    # Install ffmpeg for video encoding
    conda install -c conda-forge -y ffmpeg
    conda install -c conda-forge -y libiconv

    # Install Isaac Gym
    cd /home/admin123/Downloads/isaacgym/python
    $ENV_ROOT/bin/pip install -e .
    touch $ISAACGYM_SENTINEL
  fi

  if [[ ! -f $HOLOSOMA_SENTINEL ]]; then
    # Install Holosoma
    pip install -U pip
    pip install -e $ROOT_DIR/src/holosoma[unitree,booster]
    touch $HOLOSOMA_SENTINEL
  fi

  touch $SENTINEL_FILE
fi
