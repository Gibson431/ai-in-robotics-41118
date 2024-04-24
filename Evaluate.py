from fsae.envs import *
import time

env = RandomTrackEnv(render_mode="tp_camera")

while True:
    time.sleep(1)
