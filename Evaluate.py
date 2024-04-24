from fsae.envs import *
import time

env = RandomTrackEnv(render_mode="tp_camera")

done = False
while not done:
    action = env.action_space.sample()
    state_, reward, done, _info = env.step(action)
    _ = env.render("fp_camera")

env.close()
