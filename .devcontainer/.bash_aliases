alias sb="source ~/.bash_aliases"
alias cw="cd ~/catkin_ws"

sr () {
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    source "${HOME}/catkin_ws/devel/setup.bash"
};

cm  () {
    cw
    catkin_make -DCMAKE_EXPORT_COMPILE_COMMANDS=1
    sr
};

cm
sr