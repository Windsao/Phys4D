











input_arg=$1
USER_ID=$(id -g "$USER")


if [ -z "$input_arg" ]; then
    arch=$(uname -m)
    echo "Argument empty, trying to build based on architecture"
    if [ "$arch" == "x86_64" ]; then
        input_arg="x86"
    elif [ "$arch" == "arm64" ]; then
        input_arg="aarch64"
    elif [ "$arch" == "aarch64" ]; then
        input_arg="aarch64"
    fi
fi

user_dockerfile=user.dockerfile

if [[ $input_arg == *isaac_sim* ]] ; then
    user_dockerfile=user_isaac_sim.dockerfile
fi

echo $input_arg
echo $USER_ID

docker build --build-arg USERNAME=$USER --build-arg USER_ID=${USER_ID} --build-arg IMAGE_TAG=$input_arg -f $user_dockerfile --tag curobo_docker:user_$input_arg .