











set -e


tabs 4





export TACEX_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"






is_docker() {
    [ -f /.dockerenv ] || \
    grep -q docker /proc/1/cgroup || \
    [[ $(cat /proc/1/comm) == "containerd-shim" ]] || \
    grep -q docker /proc/mounts || \
    [[ "$(hostname)" == *"."* ]]
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
        echo -e "\t1. Conda environment is not activated." >&2
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


install_isaaclab_extension() {

    python_exe=$(extract_python_exe)

    if [ -f "$1/setup.py" ]; then
        echo -e "\t module: $1"
        ${python_exe} -m pip install --editable $1 -v
    fi
}


setup_conda_env() {

    local env_name=$1

    if ! command -v conda &> /dev/null
    then
        echo "[ERROR] Conda could not be found. Please install conda and try again."
        exit 1
    fi


    if { conda env list | grep -w ${env_name}; } >/dev/null 2>&1; then
        echo -e "[INFO] Conda environment named '${env_name}' already exists."
    else
        echo -e "[INFO] Creating conda environment named '${env_name}'..."
        conda create -y --name ${env_name} python=3.10
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


    echo -e "[INFO] Installing extra dependencies (this might take a few minutes)..."
    conda install -c conda-forge -y importlib_metadata &> /dev/null


    conda deactivate

    echo -e "[INFO] Added 'isaaclab' alias to conda environment for 'isaaclab.sh' script."
    echo -e "[INFO] Created conda environment named '${env_name}'.\n"
    echo -e "\t\t1. To activate the environment, run:                conda activate ${env_name}"
    echo -e "\t\t2. To install Isaac Lab extensions, run:            isaaclab -i"
    echo -e "\t\t4. To perform formatting, run:                      isaaclab -f"
    echo -e "\t\t5. To deactivate the environment, run:              conda deactivate"
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
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-c] -- Utility to manage Isaac Lab."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help             Display the help content."
    echo -e "\t-i, --install [all]    Install the TacEx core packages. Use 'all' to also install all extra packages [tacex_uipc]."
    echo -e "\t-f, --format           Run pre-commit to format the code and check lints."
    echo -e "\t-p, --python           Run the python executable provided by Isaac Sim or virtual environment (if active)."
    echo -e "\t-s, --sim              Run the simulator executable (isaac-sim.sh) provided by Isaac Sim."
    echo -e "\t-t, --test             Run all python unittest tests."
    echo -e "\t-o, --docker           Run the docker container helper script (docker/container.sh)."
    echo -e "\t-v, --vscode           Generate the VSCode settings file from template."
    echo -e "\t-d, --docs             Build the documentation from source using sphinx."

    echo -e "\n" >&2
}







if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 1
fi


while [[ $

    case "$1" in
        -i|--install)

            echo "[INFO] Installing extensions inside the TacEx repository..."
            python_exe=$(extract_python_exe)


            export -f extract_python_exe
            export -f install_isaaclab_extension



            echo "[INFO] Installing package [tacex]..."
            ${python_exe} -m pip install -e ${TACEX_PATH}/source/tacex

            echo "[INFO] Installing package [tacex_assets]..."
            ${python_exe} -m pip install -e ${TACEX_PATH}/source/tacex_assets

            echo "[INFO] Installing package [tacex_tasks]..."
            ${python_exe} -m pip install -e ${TACEX_PATH}/source/tacex_tasks

            if [ -z "$2" ]; then
                echo "[INFO] No extra packages installed."
            elif [ "$2" = "all" ]; then
                echo "[INFO] Installing package tacex_uipc..."
                ${python_exe} -m pip install -e ${TACEX_PATH}/source/tacex_uipc -v

                shift




            fi












            unset extract_python_exe
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
        -f|--format)



            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                cache_pythonpath=${PYTHONPATH}
                export PYTHONPATH=""
            fi


            if ! command -v pre-commit &>/dev/null; then
                echo "[INFO] Installing pre-commit..."
                pip install pre-commit
            fi

            echo "[INFO] Formatting the repository..."
            cd ${TACEX_PATH}
            pre-commit run --all-files
            cd - > /dev/null

            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                export PYTHONPATH=${cache_pythonpath}
            fi
            shift

            break
            ;;
        -p|--python)

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
            ${isaacsim_exe} --ext-folder ${TACEX_PATH}/source $@

            break
            ;;
        -t|--test)

            python_exe=$(extract_python_exe)
            shift
            ${python_exe} -m pytest ${TACEX_PATH}/tools $@

            break
            ;;
        -o|--docker)

            echo "[INFO] Running docker utility script from: ${TACEX_PATH}/docker/container.py"
            shift

            python3 "${TACEX_PATH}/docker/container.py" "${@:1}"

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

            cd ${TACEX_PATH}/docs
            ${python_exe} -m pip install -r requirements.txt > /dev/null

            ${python_exe} -m sphinx -b html -d _build/doctrees . _build/current

            echo -e "[INFO] To open documentation on default browser, run:"
            echo -e "\n\t\txdg-open $(pwd)/_build/current/index.html\n"

            cd - > /dev/null
            shift

            break
            ;;
        -h|--help)
            print_help
            exit 1
            ;;
        *)
            echo "[Error] Invalid argument provided: $1"
            print_help
            exit 1
            ;;
    esac
done
