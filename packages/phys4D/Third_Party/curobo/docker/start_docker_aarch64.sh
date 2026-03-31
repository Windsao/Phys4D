










docker run --rm -it \
--runtime nvidia \
--mount type=bind,src=/home/$USER/code,target=/home/$USER/code \
--hostname ros1-docker \
--add-host ros1-docker:127.0.0.1 \
--network host \
--gpus all \
--env ROS_HOSTNAME=localhost \
--env DISPLAY=$DISPLAY \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--volume /dev/input:/dev/input \
curobo_docker:aarch64
