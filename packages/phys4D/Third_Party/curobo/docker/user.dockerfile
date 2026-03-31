











ARG IMAGE_TAG
FROM curobo_docker:${IMAGE_TAG}

ARG USERNAME
ARG USER_ID
ARG CACHE_DATE=2024-07-19






RUN useradd -l -u $USER_ID -g users $USERNAME

RUN /sbin/adduser $USERNAME sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers




USER $USERNAME
WORKDIR /home/$USERNAME
ENV USER=$USERNAME
ENV PATH="${PATH}:/home/${USER}/.local/bin"


RUN echo 'completed'
