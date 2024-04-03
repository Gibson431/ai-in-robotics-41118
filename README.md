# ai-in-robotics-41118 #

## Dependencies ## 

- docker
- vscode
    - vscode extension: devcontainers

## Usage ##

`xhost +` in terminal

Open folder in vscode

`> Dev Container: Rebuild and Reopen in container`

Open terminal

```bash
roscore
roslaunch gazebo_tf ugv_a3.launch
rosrun ai_sim ai_sim_sample
```