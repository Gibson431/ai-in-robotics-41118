from fsae.envs import *
import time

# env = RandomTrackEnv(seed=0)
env = RandomTrackEnv(render_mode="tp_camera", seed=0)

done = False
total_reward = 0
while not done:
    action = env.action_space.sample()
    state_, reward, done, _info = env.step(action)
    print(state_)
    total_reward += reward
    # print(f"total: {total_reward}, step: {reward}")
    _ = env.render("fp_camera")

env.close()
