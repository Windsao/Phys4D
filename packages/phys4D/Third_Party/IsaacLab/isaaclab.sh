











set -e


tabs 4


export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"






install_system_deps() {

    if command -v cmake &> /dev/null; then
        echo "[INFO] cmake is already installed."
    else

        if [ "$EUID" -ne 0 ]; then
            echo "[INFO] Installing system dependencies..."
            sudo apt-get update && sudo apt-get install -y --no-install-recommends \
                cmake \
                build-essential
        else
            echo "[INFO] Installing system dependencies..."
            apt-get update && apt-get install -y --no-install-recommends \
                cmake \
                build-essential
        fi
    fi
}



is_isaacsim_version_4_5() {
    local version=""
    local python_exe
    python_exe=$(extract_python_exe)



    if [[ -f "${ISAACLAB_PATH}/_isaac_sim/VERSION" ]]; then

        version=$(head -n1 "${ISAACLAB_PATH}/_isaac_sim/VERSION" || true)
    fi



    if [[ -z "$version" ]]; then
        local sim_file=""

        sim_file=$("${python_exe}" -c 'import isaacsim, os; print(isaacsim.__file__)' 2>/dev/null || true)
        if [[ -n "$sim_file" ]]; then
            local version_path
            version_path="$(dirname "$sim_file")/../../VERSION"

            [[ -f "$version_path" ]] && version=$(head -n1 "$version_path" || true)
        fi
    fi


    if [[ -z "$version" ]]; then
        version=$("${python_exe}" <<'PY' 2>/dev/null || true
from importlib.metadata import version, PackageNotFoundError
try:
    print(version("isaacsim"))
except PackageNotFoundError:
    pass
PY
)
    fi


    [[ "$version" == 4.5* ]]
}


is_docker() {
    [ -f /.dockerenv ] || \
    grep -q docker /proc/1/cgroup || \
    [[ $(cat /proc/1/comm) == "containerd-shim" ]] || \
    grep -q docker /proc/mounts || \
    [[ "$(hostname)" == *"."* ]]
}


is_arm() {
    [[ "$(uname -m)" == "aarch64" ]] || [[ "$(uname -m)" == "arm64" ]]
}

ensure_cuda_torch() {
    local py="$1"


    local base_index="https://download.pytorch.org/whl"


    local torch_ver tv_ver cuda_ver
    if is_arm; then
        torch_ver="2.9.0"
        tv_ver="0.24.0"
        cuda_ver="130"
    else
        torch_ver="2.7.0"
        tv_ver="0.22.0"
        cuda_ver="128"
    fi

    local index="${base_index}/cu${cuda_ver}"
    local want_torch="${torch_ver}+cu${cuda_ver}"


    local cur=""
    if "$py" -m pip show torch >/dev/null 2>&1; then
        cur="$("$py" -m pip show torch 2>/dev/null | awk -F': ' '/^Version/{print $2}')"
    fi


    if [[ "$cur" == "$want_torch" ]]; then
        return 0
    fi


    echo "[INFO] Installing torch==${torch_ver} and torchvision==${tv_ver} (cu${cuda_ver}) from ${index}..."
    "$py" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
    "$py" -m pip install -U --index-url "${index}" "torch==${torch_ver}" "torchvision==${tv_ver}"
}


extract_isaacsim_path() {

    local isaac_path=${ISAACLAB_PATH}/_isaac_sim

    if [ ! -d "${isaac_path}" ]; then

        local python_exe=$(extract_python_exe)

        if [ $(${python_exe} -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            local isaac_path=$(${python_exe} -c "import isaacsim; import os; print(os.environ['ISAAC_PATH'])")
        fi
    fi

    if [ ! -d "${isaac_path}" ]; then

        echo -e "[ERROR] Unable to find the Isaac Sim directory: '${isaac_path}'" >&2
        echo -e "\tThis could be due to the following reasons:" >&2
        echo -e "\t1. Conda environment is not activated." >&2
        echo -e "\t2. Isaac Sim pip package 'isaacsim-rl' is not installed." >&2
        echo -e "\t3. Isaac Sim directory is not available at the default path: ${ISAACLAB_PATH}/_isaac_sim" >&2

        exit 1
    fi

    echo ${isaac_path}
}


extract_python_exe() {

    if ! [[ -z "${CONDA_PREFIX}" ]]; then

        local python_exe=${CONDA_PREFIX}/bin/python
    elif ! [[ -z "${VIRTUAL_ENV}" ]]; then

        local python_exe=${VIRTUAL_ENV}/bin/python
    else

        local python_exe=${ISAACLAB_PATH}/_isaac_sim/python.sh

    if [ ! -f "${python_exe}" ]; then



            if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
                local python_exe=$(which python)
            fi
        fi
    fi

    if [ ! -f "${python_exe}" ]; then
        echo -e "[ERROR] Unable to find any Python executable at path: '${python_exe}'" >&2
        echo -e "\tThis could be due to the following reasons:" >&2
        echo -e "\t1. Conda or uv environment is not activated." >&2
        echo -e "\t2. Isaac Sim pip package 'isaacsim-rl' is not installed." >&2
        echo -e "\t3. Python executable is not available at the default path: ${ISAACLAB_PATH}/_isaac_sim/python.sh" >&2
        exit 1
    fi

    echo ${python_exe}
}


extract_isaacsim_exe() {

    local isaac_path=$(extract_isaacsim_path)

    local isaacsim_exe=${isaac_path}/isaac-sim.sh

    if [ ! -f "${isaacsim_exe}" ]; then



        if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then

            local isaacsim_exe="isaacsim isaacsim.exp.full"
        else
            echo "[ERROR] No Isaac Sim executable found at path: ${isaac_path}" >&2
            exit 1
        fi
    fi

    echo ${isaacsim_exe}
}


extract_pip_command() {

    if [ -n "${VIRTUAL_ENV}" ] && [ -f "${VIRTUAL_ENV}/pyvenv.cfg" ] && grep -q "uv" "${VIRTUAL_ENV}/pyvenv.cfg"; then
        pip_command="uv pip install"
    else

        python_exe=$(extract_python_exe)
        pip_command="${python_exe} -m pip install"
    fi

    echo ${pip_command}
}

extract_pip_uninstall_command() {

    if [ -n "${VIRTUAL_ENV}" ] && [ -f "${VIRTUAL_ENV}/pyvenv.cfg" ] && grep -q "uv" "${VIRTUAL_ENV}/pyvenv.cfg"; then
        pip_uninstall_command="uv pip uninstall"
    else

        python_exe=$(extract_python_exe)
        pip_uninstall_command="${python_exe} -m pip uninstall -y"
    fi

    echo ${pip_uninstall_command}
}


install_isaaclab_extension() {

    python_exe=$(extract_python_exe)
    pip_command=$(extract_pip_command)


    if [ -f "$1/setup.py" ]; then
        echo -e "\t module: $1"
        $pip_command --editable "$1"
    fi
}


write_torch_gomp_hooks() {
  mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"


  cat > "${CONDA_PREFIX}/etc/conda/activate.d/torch_gomp.sh" <<'EOS'

: "${_IL_PREV_LD_PRELOAD:=${LD_PRELOAD-}}"

__gomp="$("$CONDA_PREFIX/bin/python" - <<'PY' 2>/dev/null || true
import pathlib
try:
    import torch
    p = pathlib.Path(torch.__file__).parent / 'lib' / 'libgomp.so.1'
    print(p if p.exists() else "", end="")
except Exception:
    pass
PY
)"

if [ -n "$__gomp" ] && [ -r "$__gomp" ]; then
  case ":${LD_PRELOAD:-}:" in
    *":$__gomp:"*) : ;;
    *) export LD_PRELOAD="$__gomp${LD_PRELOAD:+:$LD_PRELOAD}";;
  esac
fi
unset __gomp
EOS


  cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/torch_gomp_unset.sh" <<'EOS'

if [ -v _IL_PREV_LD_PRELOAD ]; then
  export LD_PRELOAD="$_IL_PREV_LD_PRELOAD"
  unset _IL_PREV_LD_PRELOAD
fi
EOS
}


begin_arm_install_sandbox() {
    if is_arm && [[ -n "${LD_PRELOAD:-}" ]]; then
        export _IL_SAVED_LD_PRELOAD="$LD_PRELOAD"
        unset LD_PRELOAD
        echo "[INFO] ARM install sandbox: temporarily unsetting LD_PRELOAD for installation."
    fi

    trap 'end_arm_install_sandbox' EXIT
}

end_arm_install_sandbox() {
    if [[ -n "${_IL_SAVED_LD_PRELOAD:-}" ]]; then
        export LD_PRELOAD="$_IL_SAVED_LD_PRELOAD"
        unset _IL_SAVED_LD_PRELOAD
    fi

    trap - EXIT
}


setup_conda_env() {

    local env_name=$1

    if ! command -v conda &> /dev/null
    then
        echo "[ERROR] Conda could not be found. Please install conda and try again."
        exit 1
    fi


    if [ ! -L "${ISAACLAB_PATH}/_isaac_sim" ] && ! python -m pip list | grep -q 'isaacsim-rl'; then
        echo -e "[WARNING] _isaac_sim symlink not found at ${ISAACLAB_PATH}/_isaac_sim"
        echo -e "\tThis warning can be ignored if you plan to install Isaac Sim via pip."
        echo -e "\tIf you are using a binary installation of Isaac Sim, please ensure the symlink is created before setting up the conda environment."
    fi


    if { conda env list | grep -w ${env_name}; } >/dev/null 2>&1; then
        echo -e "[INFO] Conda environment named '${env_name}' already exists."
    else
        echo -e "[INFO] Creating conda environment named '${env_name}'..."
        echo -e "[INFO] Installing dependencies from ${ISAACLAB_PATH}/environment.yml"


        cp "${ISAACLAB_PATH}/environment.yml"{,.bak}
        if is_isaacsim_version_4_5; then
            echo "[INFO] Detected Isaac Sim 4.5 → forcing python=3.10"
            sed -i 's/^  - python=3\.11/  - python=3.10/' "${ISAACLAB_PATH}/environment.yml"
        else
            echo "[INFO] Isaac Sim >= 5.0 detected, installing python=3.11"
        fi

        conda env create -y --file ${ISAACLAB_PATH}/environment.yml -n ${env_name}

        if [[ -f "${ISAACLAB_PATH}/environment.yml.bak" ]]; then
            mv "${ISAACLAB_PATH}/environment.yml.bak" "${ISAACLAB_PATH}/environment.yml"
        fi
    fi


    cache_pythonpath=$PYTHONPATH
    cache_ld_library_path=$LD_LIBRARY_PATH

    rm -f ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    rm -f ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh

    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}

    mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
    mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d


    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for Isaac Lab' \
        'export ISAACLAB_PATH='${ISAACLAB_PATH}'' \
        'alias isaaclab='${ISAACLAB_PATH}'/isaaclab.sh' \
        '' \
        '# show icon if not running headless' \
        'export RESOURCE_NAME="IsaacSim"' \
        '' > ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh

    write_torch_gomp_hooks


    local isaacsim_setup_conda_env_script=${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh

    if [ -f "${isaacsim_setup_conda_env_script}" ]; then

        printf '%s\n' \
            '# for Isaac Sim' \
            'source '${isaacsim_setup_conda_env_script}'' \
            '' >> ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    fi



    conda activate ${env_name}


    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for Isaac Lab' \
        'unalias isaaclab &>/dev/null' \
        'unset ISAACLAB_PATH' \
        '' \
        '# restore paths' \
        'export PYTHONPATH='${cache_pythonpath}'' \
        'export LD_LIBRARY_PATH='${cache_ld_library_path}'' \
        '' \
        '# for Isaac Sim' \
        'unset RESOURCE_NAME' \
        '' > ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh


    if [ -f "${isaacsim_setup_conda_env_script}" ]; then

        printf '%s\n' \
            '# for Isaac Sim' \
            'unset CARB_APP_PATH' \
            'unset EXP_PATH' \
            'unset ISAAC_PATH' \
            '' >> ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    fi


    conda deactivate

    echo -e "[INFO] Added 'isaaclab' alias to conda environment for 'isaaclab.sh' script."
    echo -e "[INFO] Created conda environment named '${env_name}'.\n"
    echo -e "\t\t1. To activate the environment, run:                conda activate ${env_name}"
    echo -e "\t\t2. To install Isaac Lab extensions, run:            isaaclab -i"
    echo -e "\t\t3. To perform formatting, run:                      isaaclab -f"
    echo -e "\t\t4. To deactivate the environment, run:              conda deactivate"
    echo -e "\n"
}


setup_uv_env() {

    local env_name="$1"
    local python_path="$2"


    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv could not be found. Please install uv and try again."
        echo "[ERROR] uv can be installed here:"
        echo "[ERROR] https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi


    if [ ! -L "${ISAACLAB_PATH}/_isaac_sim" ] && ! python -m pip list | grep -q 'isaacsim-rl'; then
        echo -e "[WARNING] _isaac_sim symlink not found at ${ISAACLAB_PATH}/_isaac_sim"
        echo -e "\tThis warning can be ignored if you plan to install Isaac Sim via pip."
        echo -e "\tIf you are using a binary installation of Isaac Sim, please ensure the symlink is created before setting up the conda environment."
    fi


    local env_path="${ISAACLAB_PATH}/${env_name}"
    if [ ! -d "${env_path}" ]; then
        echo -e "[INFO] Creating uv environment named '${env_name}'..."
        uv venv --clear --python "${python_path}" "${env_path}"
    else
        echo "[INFO] uv environment '${env_name}' already exists."
    fi


    local isaaclab_root="${ISAACLAB_PATH}"


    cache_pythonpath=$PYTHONPATH
    cache_ld_library_path=$LD_LIBRARY_PATH


    touch "${env_path}/bin/activate"


    cat >> "${env_path}/bin/activate" <<EOF
export ISAACLAB_PATH="${ISAACLAB_PATH}"
alias isaaclab="${ISAACLAB_PATH}/isaaclab.sh"
export RESOURCE_NAME="IsaacSim"

if [ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ]; then
    . "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh"
fi
EOF


    echo -e "[INFO] Added 'isaaclab' alias to uv environment for 'isaaclab.sh' script."
    echo -e "[INFO] Created uv environment named '${env_name}'.\n"
    echo -e "\t\t1. To activate the environment, run:                source ${env_name}/bin/activate."
    echo -e "\t\t2. To install Isaac Lab extensions, run:            isaaclab -i"
    echo -e "\t\t3. To perform formatting, run:                      isaaclab -f"
    echo -e "\t\t4. To deactivate the environment, run:              deactivate"
    echo -e "\n"
}



update_vscode_settings() {
    echo "[INFO] Setting up vscode settings..."

    python_exe=$(extract_python_exe)

    setup_vscode_script="${ISAACLAB_PATH}/.vscode/tools/setup_vscode.py"

    if [ -f "${setup_vscode_script}" ]; then
        ${python_exe} "${setup_vscode_script}"
    else
        echo "[WARNING] Unable to find the script 'setup_vscode.py'. Aborting vscode settings setup."
    fi
}


print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-n] [-c] [-u] -- Utility to manage Isaac Lab."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help           Display the help content."
    echo -e "\t-i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks as extra dependencies. Default is 'all'."
    echo -e "\t-f, --format         Run pre-commit to format the code and check lints."
    echo -e "\t-p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active)."
    echo -e "\t-s, --sim            Run the simulator executable (isaac-sim.sh) provided by Isaac Sim."
    echo -e "\t-t, --test           Run all python pytest tests."
    echo -e "\t-o, --docker         Run the docker container helper script (docker/container.sh)."
    echo -e "\t-v, --vscode         Generate the VSCode settings file from template."
    echo -e "\t-d, --docs           Build the documentation from source using sphinx."
    echo -e "\t-n, --new            Create a new external project or internal task from template."
    echo -e "\t-c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'env_isaaclab'."
    echo -e "\t-u, --uv [NAME]      Create the uv environment for Isaac Lab. Default name is 'env_isaaclab'."
    echo -e "\n" >&2
}







if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 0
fi


while [[ $

    case "$1" in
        -i|--install)

            install_system_deps

            echo "[INFO] Installing extensions inside the Isaac Lab repository..."
            python_exe=$(extract_python_exe)
            pip_command=$(extract_pip_command)
            pip_uninstall_command=$(extract_pip_uninstall_command)



            begin_arm_install_sandbox





            export -f extract_python_exe
            export -f extract_pip_command
            export -f extract_pip_uninstall_command
            export -f install_isaaclab_extension

            find -L "${ISAACLAB_PATH}/source" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_isaaclab_extension "{}"' \;

            echo "[INFO] Installing extra requirements such as learning frameworks..."

            if [ -z "$2" ]; then
                echo "[INFO] Installing all rl-frameworks..."
                framework_name="all"
            elif [ "$2" = "none" ]; then
                echo "[INFO] No rl-framework will be installed."
                framework_name="none"
                shift
            else
                echo "[INFO] Installing rl-framework: $2"
                framework_name=$2
                shift
            fi

            ${pip_command} -e "${ISAACLAB_PATH}/source/isaaclab_rl[${framework_name}]"
            ${pip_command} -e "${ISAACLAB_PATH}/source/isaaclab_mimic[${framework_name}]"






            end_arm_install_sandbox



            if is_docker; then
                echo "[INFO] Running inside a docker container. Skipping VSCode settings setup."
                echo "[INFO] To setup VSCode settings, run 'isaaclab -v'."
            else

                update_vscode_settings
            fi


            unset extract_python_exe
            unset extract_pip_command
            unset extract_pip_uninstall_command
            unset install_isaaclab_extension
            shift
            ;;
        -c|--conda)

            if [ -z "$2" ]; then
                echo "[INFO] Using default conda environment name: env_isaaclab"
                conda_env_name="env_isaaclab"
            else
                echo "[INFO] Using conda environment name: $2"
                conda_env_name=$2
                shift
            fi

            setup_conda_env ${conda_env_name}
            shift
            ;;
        -u|--uv)

            if [ -z "$2" ]; then
                echo "[INFO] Using default uv environment name: env_isaaclab"
                uv_env_name="env_isaaclab"
            else
                echo "[INFO] Using uv environment name: $2"
                uv_env_name=$2
                shift
            fi

            setup_uv_env ${uv_env_name}
            shift
            ;;
        -f|--format)



            if [ -n "${CONDA_DEFAULT_ENV}" ] || [ -n "${VIRTUAL_ENV}" ]; then
                cache_pythonpath=${PYTHONPATH}
                export PYTHONPATH=""
            fi


            if ! command -v pre-commit &>/dev/null; then
                echo "[INFO] Installing pre-commit..."
                pip_command=$(extract_pip_command)
                ${pip_command} pre-commit
                sudo apt-get install -y pre-commit
            fi

            echo "[INFO] Formatting the repository..."
            cd ${ISAACLAB_PATH}
            pre-commit run --all-files
            cd - > /dev/null

            if [ -n "${CONDA_DEFAULT_ENV}" ] || [ -n "${VIRTUAL_ENV}" ]; then
                export PYTHONPATH=${cache_pythonpath}
            fi

            shift

            break
            ;;
        -p|--python)

            if is_arm; then
                export RESOURCE_NAME="${RESOURCE_NAME:-IsaacSim}"
            fi

            python_exe=$(extract_python_exe)
            echo "[INFO] Using python from: ${python_exe}"
            shift
            ${python_exe} "$@"

            break
            ;;
        -s|--sim)

            isaacsim_exe=$(extract_isaacsim_exe)
            echo "[INFO] Running isaac-sim from: ${isaacsim_exe}"
            shift
            ${isaacsim_exe} --ext-folder ${ISAACLAB_PATH}/source $@

            break
            ;;
        -n|--new)

            python_exe=$(extract_python_exe)
            pip_command=$(extract_pip_command)
            shift
            echo "[INFO] Installing template dependencies..."
            ${pip_command} -q -r ${ISAACLAB_PATH}/tools/template/requirements.txt
            echo -e "\n[INFO] Running template generator...\n"
            ${python_exe} ${ISAACLAB_PATH}/tools/template/cli.py $@

            break
            ;;
        -t|--test)

            python_exe=$(extract_python_exe)
            shift
            ${python_exe} -m pytest ${ISAACLAB_PATH}/tools $@

            break
            ;;
        -o|--docker)

            docker_script=${ISAACLAB_PATH}/docker/container.sh
            echo "[INFO] Running docker utility script from: ${docker_script}"
            shift
            bash ${docker_script} $@

            break
            ;;
        -v|--vscode)

            update_vscode_settings
            shift

            break
            ;;
        -d|--docs)

            echo "[INFO] Building documentation..."

            python_exe=$(extract_python_exe)
            pip_command=$(extract_pip_command)

            cd ${ISAACLAB_PATH}/docs
            ${pip_command} -r requirements.txt > /dev/null

            ${python_exe} -m sphinx -b html -d _build/doctrees . _build/current

            echo -e "[INFO] To open documentation on default browser, run:"
            echo -e "\n\t\txdg-open $(pwd)/_build/current/index.html\n"

            cd - > /dev/null
            shift

            break
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "[Error] Invalid argument provided: $1"
            print_help
            exit 1
            ;;
    esac
done
