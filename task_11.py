import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import your environment
from ot2_env_wrapper import OT2Env 

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "./best_model.zip" 
START_POS = [-0.150, -0.150, 0.250] 
TARGET_POS = [ 0.200,  0.150, 0.170]

SIM_TIMESTEP = 1.0 / 240.0
MAX_STEPS = 1000

# Constants for the stop condition
POS_THRESHOLD = 0.001   # 1mm
VEL_THRESHOLD = 0.01    # Rad/s

# ==========================================
# 2. HELPER: TELEPORT
# ==========================================
def force_reset_environment(env, start_pos, target_pos):
    obs, _ = env.reset()
    env.goal_position = np.array(target_pos, dtype=np.float32)
    obs[3:6] = env.goal_position
    return obs

# ==========================================
# 3. MAIN LOOP
# ==========================================
def run_ppo_test():
    # Render=True if you want to see the window, False if headless
    env = OT2Env(render=True, max_steps=MAX_STEPS)
    
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print("❌ Model file not found! Please check MODEL_PATH.")
        return

    obs = force_reset_environment(env, START_POS, TARGET_POS)
    
    # --- DATA STORAGE ---
    history = {
        'time': [],
        'target_x': [], 'actual_x': [], 'error_x': [],
        'target_y': [], 'actual_y': [], 'error_y': [],
        'target_z': [], 'actual_z': [], 'error_z': []
    }
    
    print("Starting PPO Inference...")
    current_time = 0.0
    
    for i in range(MAX_STEPS):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract Data from Observation
        # obs structure: [px, py, pz, gx, gy, gz, vx, vy, vz]
        curr_pos = obs[0:3]
        velocities = obs[6:9] # vx, vy, vz
        
        # Calculate Errors
        err_x = TARGET_POS[0] - curr_pos[0]
        err_y = TARGET_POS[1] - curr_pos[1]
        err_z = TARGET_POS[2] - curr_pos[2]
        
        # --- YOUR STOP LOGIC ---
        # Calculate Euclidean distance to target
        dist = sum([(t - c)**2 for t, c in zip(TARGET_POS, curr_pos)]) ** 0.5
        
        # Check if velocities are low enough
        is_stopped = all(abs(v) < VEL_THRESHOLD for v in velocities)
        
        # If position is close and robot is stopped, break immediately
        if dist < POS_THRESHOLD and is_stopped:
            print(f"✅ Target reached at step {i} (Time: {current_time:.3f}s)")
            break
        # -----------------------

        # Log Data
        history['time'].append(current_time)
        
        history['target_x'].append(TARGET_POS[0])
        history['actual_x'].append(curr_pos[0])
        history['error_x'].append(err_x)
        
        history['target_y'].append(TARGET_POS[1])
        history['actual_y'].append(curr_pos[1])
        history['error_y'].append(err_y)
        
        history['target_z'].append(TARGET_POS[2])
        history['actual_z'].append(curr_pos[2])
        history['error_z'].append(err_z)
        
        current_time += SIM_TIMESTEP
        time.sleep(0.01) 
        
        if truncated:
            print(f"⚠️ Episode timed out at step {i}")
            break
    print(dist)
    env.close()
    
    # Save the plot to file
    save_results(history)

def save_results(h):
    # Create a 2x3 grid (2 Rows, 3 Columns)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"PPO Step Response & Error Analysis", fontsize=16)

    # --- ROW 1: POSITION TRACKING ---
    # X Axis
    axs[0, 0].plot(h['time'], h['target_x'], 'b--', label='Target')
    axs[0, 0].plot(h['time'], h['actual_x'], 'r-', label='Actual')
    axs[0, 0].set_title("X Position")
    axs[0, 0].set_ylabel("Position (m)")
    axs[0, 0].grid(True, alpha=0.5)
    axs[0, 0].legend()

    # Y Axis
    axs[0, 1].plot(h['time'], h['target_y'], 'b--', label='Target')
    axs[0, 1].plot(h['time'], h['actual_y'], 'r-', label='Actual')
    axs[0, 1].set_title("Y Position")
    axs[0, 1].grid(True, alpha=0.5)
    
    # Z Axis
    axs[0, 2].plot(h['time'], h['target_z'], 'b--', label='Target')
    axs[0, 2].plot(h['time'], h['actual_z'], 'r-', label='Actual')
    axs[0, 2].set_title("Z Position")
    axs[0, 2].grid(True, alpha=0.5)

    # --- ROW 2: ERROR TRACKING ---
    # X Error
    axs[1, 0].plot(h['time'], h['error_x'], 'k-', label='Error')
    axs[1, 0].axhline(0, color='g', linestyle='--', alpha=0.5) # Zero line
    axs[1, 0].set_title("X Error")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Error (m)")
    axs[1, 0].grid(True, alpha=0.5)

    # Y Error
    axs[1, 1].plot(h['time'], h['error_y'], 'k-', label='Error')
    axs[1, 1].axhline(0, color='g', linestyle='--', alpha=0.5)
    axs[1, 1].set_title("Y Error")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].grid(True, alpha=0.5)

    # Z Error
    axs[1, 2].plot(h['time'], h['error_z'], 'k-', label='Error')
    axs[1, 2].axhline(0, color='g', linestyle='--', alpha=0.5)
    axs[1, 2].set_title("Z Error")
    axs[1, 2].set_xlabel("Time (s)")
    axs[1, 2].grid(True, alpha=0.5)

    plt.tight_layout()
    
    # Save the figure
    filename = "ppo_results.png"
    plt.savefig(filename)
    print(f"📊 Plot saved to {filename}")

if __name__ == "__main__":
    run_ppo_test()