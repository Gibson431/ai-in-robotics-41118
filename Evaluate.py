from fsae.envs import *
from ddpg import ReplayBuffer, DDPGAgent, EpsilonGreedy

# env = RandomTrackEnv(render_mode="tp_camera", seed=0)
env = RandomTrackEnv(render_mode="detections", seed=0)

# Initialize the agent, replay buffer, and environment
state_dim = 8  # Dimension of the state space
action_dim = 2  # Dimension of the action space
hidden_dim = 256
max_action = (1, 0.6)  # Maximum value of the action
num_episodes = 1000
max_steps = 25
batch_size = 500

replay_buffer = ReplayBuffer(
    buffer_size=50000, state_dim=state_dim, action_dim=action_dim
)
agent = DDPGAgent(state_dim, action_dim, hidden_dim, replay_buffer, max_action)
agent.load_weights()

done = False
state = np.zeros(8)

total_reward = 0
counter = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

    total_reward += reward
    counter += 1

    if counter % 50 == 0:
        print(f"{counter} : total reward: {total_reward}")

print(f"{counter} : Finished simulation. Total reward: {total_reward}")
