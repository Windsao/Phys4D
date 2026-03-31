











docker run --rm -it \
--privileged \
-e NVIDIA_DISABLE_REQUIRE=1 \
-e NVIDIA_DRIVER_CAPABILITIES=all  --device /dev/dri \
--hostname ros1-docker \
--add-host ros1-docker:127.0.0.1 \
--gpus all \
--network host \
--env DISPLAY=unix$DISPLAY \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--volume /dev:/dev \
curobo_docker:x86
