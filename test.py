from ot2_env_wrapper import OT2Env
import numpy as np

env = OT2Env(render=True)

obs = env.reset()
print("Initial observation:", obs)

for step in range(10):
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
