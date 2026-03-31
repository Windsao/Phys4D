






set -e


tabs 4


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"





display_warning() {
    echo -e "\033[31mWARNING: $1\033[0m"
}


version_gte() {

    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]
}


check_docker_version() {

    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed! Please check the 'Docker Guide' for instruction." >&2;
        exit 1
    fi

    docker_version=$(docker --version | awk '{ print $3 }')
    apptainer_version=$(apptainer --version | awk '{ print $3 }')


    if [ "$docker_version" = "24.0.7" ] && [ "$apptainer_version" = "1.2.5" ]; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."


    elif version_gte "$docker_version" "27.0.0" && version_gte "$apptainer_version" "1.3.4"; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."


    else
        display_warning "Docker version ${docker_version} and Apptainer version ${apptainer_version} are non-tested versions. There could be issues, please try to update them. More info: https://isaac-sim.github.io/IsaacLab/source/deployment/cluster.html"
    fi
}


check_image_exists() {
    image_name="$1"
    if ! docker image inspect $image_name &> /dev/null; then
        echo "[Error] The '$image_name' image does not exist!" >&2;
        echo "[Error] You might be able to build it with /IsaacLab/docker/container.py." >&2;
        exit 1
    fi
}


check_singularity_image_exists() {
    image_name="$1"
    if ! ssh "$CLUSTER_LOGIN" "[ -f $CLUSTER_SIF_PATH/$image_name.tar ]"; then
        echo "[Error] The '$image_name' image does not exist on the remote host $CLUSTER_LOGIN!" >&2;
        exit 1
    fi
}

submit_job() {

    echo "[INFO] Arguments passed to job script ${@}"

    case $CLUSTER_JOB_SCHEDULER in
        "SLURM")
            job_script_file=submit_job_slurm.sh
            ;;
        "PBS")
            job_script_file=submit_job_pbs.sh
            ;;
        *)
            echo "[ERROR] Unsupported job scheduler specified: '$CLUSTER_JOB_SCHEDULER'. Supported options are: ['SLURM', 'PBS']"
            exit 1
            ;;
    esac

    ssh $CLUSTER_LOGIN "cd $CLUSTER_ISAACLAB_DIR && bash $CLUSTER_ISAACLAB_DIR/docker/cluster/$job_script_file \"$CLUSTER_ISAACLAB_DIR\" \"isaac-lab-$profile\" ${@}"
}







help() {
    echo -e "\nusage: $(basename "$0") [-h] <command> [<profile>] [<job_args>...] -- Utility for interfacing between IsaacLab and compute clusters."
    echo -e "\noptions:"
    echo -e "  -h              Display this help message."
    echo -e "\ncommands:"
    echo -e "  push [<profile>]              Push the docker image to the cluster."
    echo -e "  job [<profile>] [<job_args>]  Submit a job to the cluster."
    echo -e "\nwhere:"
    echo -e "  <profile>  is the optional container profile specification. Defaults to 'base'."
    echo -e "  <job_args> are optional arguments specific to the job command."
    echo -e "\n" >&2
}


while getopts ":h" opt; do
    case ${opt} in
        h )
            help
            exit 0
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            help
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))


if [ $
    echo "Error: Command is required." >&2
    help
    exit 1
fi

command=$1
shift
profile="base"

case $command in
    push)
        if [ $
            echo "Error: Too many arguments for push command." >&2
            help
            exit 1
        fi
        [ $
        echo "Executing push command"
        [ -n "$profile" ] && echo "Using profile: $profile"
        if ! command -v apptainer &> /dev/null; then
            echo "[INFO] Exiting because apptainer was not installed"
            echo "[INFO] You may follow the installation procedure from here: https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages"
            exit
        fi

        check_image_exists isaac-lab-$profile:latest

        check_docker_version

        source $SCRIPT_DIR/.env.cluster

        mkdir -p /$SCRIPT_DIR/exports

        rm -rf /$SCRIPT_DIR/exports/isaac-lab-$profile*



        cd /$SCRIPT_DIR/exports
        APPTAINER_NOHTTPS=1 apptainer build --sandbox --fakeroot isaac-lab-$profile.sif docker-daemon:

        tar -cvf /$SCRIPT_DIR/exports/isaac-lab-$profile.tar isaac-lab-$profile.sif

        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_SIF_PATH"

        scp $SCRIPT_DIR/exports/isaac-lab-$profile.tar $CLUSTER_LOGIN:$CLUSTER_SIF_PATH/isaac-lab-$profile.tar
        ;;
    job)
        if [ $
            passed_profile=$1
            if [ -f "$SCRIPT_DIR/../.env.$passed_profile" ]; then
                profile=$passed_profile
                shift
            fi
        fi
        job_args="$@"
        echo "[INFO] Executing job command"
        [ -n "$profile" ] && echo -e "\tUsing profile: $profile"
        [ -n "$job_args" ] && echo -e "\tJob arguments: $job_args"
        source $SCRIPT_DIR/.env.cluster

        current_datetime=$(date +"%Y%m%d_%H%M%S")

        CLUSTER_ISAACLAB_DIR="${CLUSTER_ISAACLAB_DIR}_${current_datetime}"

        check_singularity_image_exists isaac-lab-$profile

        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_ISAACLAB_DIR"

        echo "[INFO] Syncing Isaac Lab code..."
        rsync -rh  --exclude="*.git*" --filter=':- .dockerignore'  /$SCRIPT_DIR/../.. $CLUSTER_LOGIN:$CLUSTER_ISAACLAB_DIR

        echo "[INFO] Executing job script..."

        submit_job $job_args
        ;;
    *)
        echo "Error: Invalid command: $command" >&2
        help
        exit 1
        ;;
esac
