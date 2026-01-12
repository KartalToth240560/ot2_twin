import time
import numpy as np
import random
import pybullet as p
from stable_baselines3 import PPO

# Import your custom environment and components
from ot2_gym_wrapper import OT2Env 
from task10_pid_controller.pid import PIDController
from sim_class import Simulation

# ==========================================
# 1. SHARED CONFIGURATION & CONSTANTS
# ==========================================
X_LIMITS = [-0.1870, 0.2530]
Y_LIMITS = [-0.1705, 0.2195]
Z_LIMITS = [0.1195, 0.2895]
MARGIN = 0.05
NUM_TEST_POINTS = 5

# Thresholds
POS_THRESHOLD = 0.001  # 1mm
VEL_THRESHOLD = 0.01   # Stopped velocity
TIMEOUT = 5.0          # Max seconds per point

# PID Gains
PID_GAINS = {
    'x': {'kp': 25.0, 'ki': 0.3, 'kd': 0.4},
    'y': {'kp': 25.0, 'ki': 0.3, 'kd': 0.4},
    'z': {'kp': 25.0, 'ki': 0.000, 'kd': 0.1},
}

MODEL_PATH = "./best_model.zip"

def get_random_points(n):
    points = []
    for _ in range(n):
        points.append([
            random.uniform(X_LIMITS[0] + MARGIN, X_LIMITS[1] - MARGIN),
            random.uniform(Y_LIMITS[0] + MARGIN, Y_LIMITS[1] - MARGIN),
            random.uniform(Z_LIMITS[0] + MARGIN, Z_LIMITS[1] - MARGIN)
        ])
    return points

# Helper for PID world-to-joint conversion
def calculate_joint_targets(robotId, target_pos, pipette_offset):
    base_pos = list(p.getBasePositionAndOrientation(robotId)[0])
    return [base_pos[0] + pipette_offset[0] - target_pos[0],
            base_pos[1] + pipette_offset[1] - target_pos[1],
            target_pos[2] - base_pos[2] - pipette_offset[2]]

def get_pipette_pos(robotId, pipette_offset):
    s_x = p.getJointState(robotId, 0)[0]
    s_y = p.getJointState(robotId, 1)[0]
    s_z = p.getJointState(robotId, 2)[0]
    base_pos = list(p.getBasePositionAndOrientation(robotId)[0])
    return [base_pos[0] - s_x + pipette_offset[0], 
            base_pos[1] - s_y + pipette_offset[1], 
            base_pos[2] + s_z + pipette_offset[2]]

# ==========================================
# 2. EVALUATION FUNCTIONS
# ==========================================

def run_pid_eval(points):
    print("\n--- Starting PID Evaluation ---")
    sim = Simulation(num_agents=1, render=False) # Set render=True if preferred
    robotId = sim.robotIds[0]
    results = []

    pid_x = PIDController(**PID_GAINS['x'], output_limits=(-1.0, 1.0))
    pid_y = PIDController(**PID_GAINS['y'], output_limits=(-1.0, 1.0))
    pid_z = PIDController(**PID_GAINS['z'], output_limits=(-1.0, 1.0))

    for i, target in enumerate(points):
        pid_x.reset(); pid_y.reset(); pid_z.reset()
        target_joints = calculate_joint_targets(robotId, target, sim.pipette_offset)
        
        start_time = time.time()
        reached = False
        final_err = 0
        
        while (time.time() - start_time) < TIMEOUT:
            s_x = p.getJointState(robotId, 0)
            s_y = p.getJointState(robotId, 1)
            s_z = p.getJointState(robotId, 2)
            
            vx = pid_x.update(target_joints[0], s_x[0], 1./240.)
            vy = pid_y.update(target_joints[1], s_y[0], 1./240.)
            vz = pid_z.update(target_joints[2], s_z[0], 1./240.)
            
            sim.run([[-vx, -vy, vz, 0]], num_steps=1)
            
            curr_pos = get_pipette_pos(robotId, sim.pipette_offset)
            dist = np.linalg.norm(np.array(target) - np.array(curr_pos))
            vels = [s_x[1], s_y[1], s_z[1]]
            
            if dist < POS_THRESHOLD and all(abs(v) < VEL_THRESHOLD for v in vels):
                reached = True
                final_err = dist
                break
            final_err = dist

        duration = time.time() - start_time
        results.append({'time': duration, 'error': final_err})
        print(f"Point {i+1} Done.")

    p.disconnect()
    return results

def run_ppo_eval(points):
    print("\n--- Starting PPO Evaluation ---")
    env = OT2Env(render=False, max_steps=2000)
    model = PPO.load(MODEL_PATH)
    results = []

    for i, target in enumerate(points):
        obs, _ = env.reset()
        env.goal_position = np.array(target, dtype=np.float32)
        obs[3:6] = env.goal_position # Inject target into observation
        
        start_time = time.time()
        final_err = 0
        
        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            curr_pos = obs[0:3]
            dist = np.linalg.norm(np.array(target) - np.array(curr_pos))
            vels = obs[6:9]

            if dist < POS_THRESHOLD and all(abs(v) < VEL_THRESHOLD for v in vels):
                break
            
            final_err = dist
            if terminated or truncated: break

        duration = time.time() - start_time
        results.append({'time': duration, 'error': final_err})
        print(f"Point {i+1} Done.")

    env.close()
    return results

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    test_points = get_random_points(NUM_TEST_POINTS)
    
    pid_res = run_pid_eval(test_points)
    ppo_res = run_ppo_eval(test_points)

    print("\n" + "="*50)
    print(f"{'Pt':<4} | {'PID Time':<10} | {'PID Error (mm)':<15} | {'PPO Time':<10} | {'PPO Error (mm)':<15}")
    print("-" * 75)
    
    for i in range(NUM_TEST_POINTS):
        print(f"{i+1:<4} | "
              f"{pid_res[i]['time']:<10.3f} | "
              f"{pid_res[i]['error']*1000:<15.4f} | "
              f"{ppo_res[i]['time']:<10.3f} | "
              f"{ppo_res[i]['error']*1000:<15.4f}")
    
    # Calculate Averages
    avg_pid_t = np.mean([r['time'] for r in pid_res])
    avg_ppo_t = np.mean([r['time'] for r in ppo_res])
    avg_pid_e = np.mean([r['error'] for r in pid_res]) * 1000
    avg_ppo_e = np.mean([r['error'] for r in ppo_res]) * 1000

    print("-" * 75)
    print(f"{'AVG':<4} | {avg_pid_t:<10.3f} | {avg_pid_e:<15.4f} | {avg_ppo_t:<10.3f} | {avg_ppo_e:<15.4f}")
    print("="*50)