import msvcrt
from fsae.envs import RandomTrackEnv
import time
from ddpg import ReplayBuffer, DDPGAgent
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

time_get_key = 0
time_main = 0

def get_key(): 
    key = None
    start_time = time.time()
    # Check if there's data available to read from sys.stdin without blocking
    if msvcrt.kbhit():
        key = msvcrt.getch().decode()

    end_time = time.time()
    global time_get_key
    time_get_key += end_time - start_time

    return key

# Initialize the agent, replay buffer, and environment
state_dim = 12 # Dimension of the state space
action_dim = 2 # Dimension of the action space
hidden_dim = 256
max_action = 0.6 # Maximum value of the action
num_episodes = 10000
max_steps = 1000
batch_size = 128

episode_history = []
episode_reward_history = []
replay_buffer = ReplayBuffer(buffer_size=1000000, state_dim=state_dim, action_dim=action_dim)
agent = DDPGAgent(state_dim, action_dim, hidden_dim, replay_buffer, max_action)

model_file = "policy_network_e.pkl"


def main():
    env = RandomTrackEnv(render_mode="tp_camera")
    _ = env.render("fp_camera")
    state = env.reset(seed=0)
    episode_reward = 0

    done = False
    print("started")
    action = [0,0]
    while not done:
        start_time_main = time.time()
        key = get_key() # This is a non-blocking call now
        if key is not None:
            if key == "q":  # Move forward-left
                action = [1, 0.6]
            elif key == "w":  # Move forward
                action = [1, 0.0]
            elif key == "e":  # Move forward-right
                action = [1, -0.6]
            elif key == "a":  # Steer left (no throttle)
                action = [0, 0.6]
            elif key == "s":  # No steer / throttle
                action = [0, 0]
            elif key == "d":  # Steer right (no throttle)
                action = [0, -0.6]
            elif key == "z":  # Move back-left
                action = [-1, 0.6]
            elif key == "x":  # Move back
                action = [-1, 0.0]
            elif key == "c":  # Move back-right
                action = [-1, -0.6]
            elif key == "p":  # quit application
                break
            print(f"{key} recognised. Action: ", action)
        #If no key is pressed, pass the same command 

        state_, reward, done, _info = env.step(action)
        agent.replay_buffer.add(state, action, reward, state_, done)
        state = state_ #dont think i need this
        episode_reward += reward

        end_time_step = time.time()
        global time_main
        time_main += end_time_step - start_time_main
        #_ = env.render("fp_camera")
        print("tick")

    if replay_buffer.buffer_size > batch_size:
            agent.train(batch_size)

    torch.save(
        agent.critic.state_dict(),
        model_file,
    )
    env.close()
    
    # Test the training

def test():
    print("starting testing sequence")

    env = RandomTrackEnv(render_mode="tp_camera")
    _ = env.render("fp_camera")
    episode_reward = 0
    done = False
    replay_buffer = ReplayBuffer(buffer_size=1000000, state_dim=state_dim, action_dim=action_dim)
    agent = DDPGAgent(state_dim, action_dim, hidden_dim, replay_buffer, max_action)
    try:
        agent.critic.load_state_dict(torch.load(model_file))
    except:
        print("------------------------ yooooo this sucks - code better ------------------------------")
        print("ending sequence")
        return
    total_reward = 0
    for i in range(5):
        state = env.reset(seed=i)
        while not done:
            with torch.no_grad():
                q_values = agent.critic(torch.tensor(state, dtype=torch.float32), [0,0])
            action = torch.argmax(q_values).item()
            state_, reward, done, _ = env.step(action)
            total_reward += reward
            print(f"total: {total_reward}, step: {reward}")
            _ = env.render("fp_camera")
            
    env.close()


if __name__ == "__main__":
    Train = False
    Test = True
    if Train:
        main()
    if Test:
        test()



print("Total time spent in get_key function:", time_get_key)
print("Total time spent in main function:", time_main)

"""
load the data back in at the start to fill up the replay buffer

use the replay buffer for states, rewards, next steps

    """