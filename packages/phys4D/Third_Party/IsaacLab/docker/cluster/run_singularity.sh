

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"





setup_directories() {

    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}








SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base


setup_directories

cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR


mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"


cp -r $1 $TMPDIR

dir_name=$(basename "$1")


tar -xf $CLUSTER_SIF_PATH/$2.tar  -C $TMPDIR




singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw \
    --nv --writable --containall $TMPDIR/$2.sif \
    bash -c "export ISAACLAB_PATH=/workspace/isaaclab && cd /workspace/isaaclab && /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}"


rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..


if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

echo "(run_singularity.py): Return"
