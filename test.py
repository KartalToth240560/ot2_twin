from ot2_env_wrapper import OT2Env
import numpy as np

env = OT2Env(render=True)
from stable_baselines3.common.env_checker import check_env


# It will raise an error or warning if your environment is not compatible
check_env(env)
print("Environment check successful!")

obs = env.reset()
print("Initial observation:", obs)

for step in range(1000):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step}")
    print("  Action:      ", action)
    print("  Observation: ", obs)
    print("  Reward:      ", reward)
    print("  Terminated:  ", terminated)
    print("  Truncated:   ", truncated)

    if terminated or truncated:
        print("Episode finished.")
        break
