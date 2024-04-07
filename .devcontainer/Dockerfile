FROM osrf/ros:noetic-desktop-full

# Add vscode user with same UID and GID as your host system
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
    
# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git
RUN sudo apt install -y git

# Customise the installation
USER root

RUN apt-get update -y && apt-get install -y \
    python3 \
    python3-yaml \
    python3-rospkg \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    gdb

RUN apt-get update -y && apt-get install -y \
    ros-noetic-navigation \
    ros-noetic-slam-toolbox \
    ros-noetic-slam-gmapping 

RUN apt-get update -y && apt-get install -y \
    ros-noetic-robot-controllers \
    ros-noetic-moveit-core \
    ros-noetic-teleop-twist-keyboard \
    ros-noetic-topic-tools

RUN apt-get update -y && apt-get install -y \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-robot-state-publisher \
    ros-noetic-tf2-ros \
    libignition-math4-dev

RUN apt-get update && apt-get install -y \
    python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python

# Switch to user
USER $USERNAME

RUN pip install defusedxml

# Rosdep update
RUN rosdep update

# Create ros workspace
RUN mkdir -p /home/$USERNAME/catkin_ws/src

# Example for cloning external git repos
RUN git clone -b noetic https://github.com/41012/pfms-support.git \
    /home/$USERNAME/Downloads/pfms-support

# RUN cd /home/$USERNAME/Downloads/pfms-support/ && git checkout noetic && cd /home/$USERNAME

RUN git clone https://github.com/mats-robotics/detection_msgs.git

RUN git clone --recurse-submodules https://github.com/mats-robotics/yolov5_ros.git 

RUN cd yolov5_ros/src/yolov5

RUN pip install -r requirements.txt

ADD https://github.com/41012/pfms-support/raw/noetic/packages/pipes-2.5.0-noetic-Linux.deb /tmp

RUN sudo dpkg -i /tmp/pipes-2.5.0-noetic-Linux.deb

RUN ln -s /home/$USERNAME/Downloads/pfms-support/pfms-ros /home/$USERNAME/catkin_ws/src/

# RUN git clone git@github.com:NovoG93/sjtu-drone.git /home/$USERNAME/catkin_ws/src/sjtu_drone

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc