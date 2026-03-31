







echo -e "\e[33m------------------------------------------------------------"
echo -e "WARNING: This script is deprecated and will be removed in the future. Please use 'docker/container.py' instead."
echo -e "------------------------------------------------------------\e[0m\n"


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


python3 "${SCRIPT_DIR}/container.py" "${@:1}"
