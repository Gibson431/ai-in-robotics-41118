from fsae.envs import *
import time
import sys

env = RandomTrackEnv(render_mode="tp_camera")

done = False
print("started")
while not done:
    key = sys.stdin.read(1)
    print(f"{key}\n")
    action = []
    if key == "q":  # Move forward-left
        action = [1, -0.6]
    elif key == "w":  # Move forward
        action = [1, 0.0]
    elif key == "e":  # Move forward-right
        action = [1, 0.6]
    elif key == "a":  # Steer left (no throttle)
        action = [0, -0.6]
    elif key == "s":  # No steer / throttle
        action = [0, 0]
    elif key == "d":  # Steer right (no throttle)
        action = [0, 0.6]
    elif key == "z":  # Move back-left
        action = [-1, -0.6]
    elif key == "x":  # Move back
        action = [-1, 0.0]
    elif key == "c":  # Move back-right
        action = [-1, 0.6]
    elif key == "p":  # quit application
        break
    else:
        key = "No input"
        action = [0, 0]

    if key != "No input":
        print(f"{key} recognised. Action: ", action)
        state_, reward, done, _info = env.step(action)
        _ = env.render("fp_camera")
    print("tick")

env.close()
