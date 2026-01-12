import os
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model

# --- Custom Modules ---
from sim_class import Simulation
from task_5.inference_single import inference
from task_5.models.simple_unet import f1
from bottom_positons import get_bottom
from task10_pid_controller.pid import PIDController

# ==========================================
# 1. CONFIGURATION
# ==========================================
PID_GAINS = {
    'x': {'kp': 25.0, 'ki': 0.005, 'kd': 0.5},
    'y': {'kp': 25.0, 'ki': 0.005, 'kd': 0.5},
    'z': {'kp': 25.0, 'ki': 0.005, 'kd': 0.1},
}

SIM_TIMESTEP = 1.0 / 240.0
SPEED_LIMIT = 3.0
POS_THRESHOLD = 0.001
VEL_THRESHOLD = 0.01 
STEADY_FRAMES = 10 

# --- ROBOT CALIBRATION ---
PLATE_X = 0.10775
PLATE_Y = 0.088 - 0.026 
PLATE_Z = 0.057
PLATE_POSITION_ROBOT = np.array([PLATE_X, PLATE_Y, PLATE_Z], dtype=np.float32)

PLATE_SIZE_MM = 150.0
WAIT_STEPS = int(0.005 / SIM_TIMESTEP) + 1 

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_joint_targets(robotId, target_pos, pipette_offset):
    base_pos = list(p.getBasePositionAndOrientation(robotId)[0])
    target_joint_x = base_pos[0] + pipette_offset[0] - target_pos[0]
    target_joint_y = base_pos[1] + pipette_offset[1] - target_pos[1]
    target_joint_z = target_pos[2] - base_pos[2] - pipette_offset[2]
    return [target_joint_x, target_joint_y, target_joint_z]

def convert_joint_to_pipette_position(robotId, joint_values, pipette_offset):
    base_pos = list(p.getBasePositionAndOrientation(robotId)[0])
    base_pos[0] -= joint_values[0]
    base_pos[1] -= joint_values[1]
    base_pos[2] += joint_values[2]
    return [base_pos[0]+pipette_offset[0], base_pos[1]+pipette_offset[1], base_pos[2]+pipette_offset[2]]

# ==========================================
# 3. MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    
    # --- PHASE 1: INFERENCE ---
    patch_size = 256
    model_path = f'../deliverables/kartaltoth_240560_unet_model_{patch_size}px.h5'
    model = load_model(model_path, custom_objects={"f1": f1})
    
    sim = Simulation(num_agents=1, render=True)
    image_path = sim.get_plate_image()
    
    print(f"\n[INFO] Processing image: {image_path}")
    
    predicted_mask, preprocess_info = inference(image_path, model, patch_size, threshold=0.3)
    root_bottoms = get_bottom(predicted_mask, preprocess_info)

    # --- PHASE 2: COORDINATE TRANSFORMATION ---
    targets_queue = []
    pixel_points = [] 
    
    img_width = preprocess_info.crop_size
    plate_size_pixels = img_width 
    conversion_factor = PLATE_SIZE_MM / plate_size_pixels
    
    for k, v in root_bottoms.items():
        if v is not None:
            pixel_points.append(v)
            y_pixel, x_pixel = v 
            x_mm = x_pixel * conversion_factor
            y_mm = y_pixel * conversion_factor
            
            dx_meter = x_mm / 1000.0
            dy_meter = y_mm / 1000.0
            
            final_x = PLATE_POSITION_ROBOT[0] + dx_meter
            final_y = PLATE_POSITION_ROBOT[1] + dy_meter
            final_z = 0.2 
            
            final_pos = np.array([final_x, final_y, final_z], dtype=np.float32)
            targets_queue.append(final_pos)

    if not targets_queue:
        print("[WARN] No roots detected. Exiting.")
        sim.close()
        exit()

    # --- PHASE 3: SIMULATION & BENCHMARKING ---
    print("\n[INFO] Starting Simulation and Performance Evaluation...")
    
    robotId = sim.robotIds[0]
    p.setTimeStep(SIM_TIMESTEP)

    pids = {
        'x': PIDController(**PID_GAINS['x'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT)),
        'y': PIDController(**PID_GAINS['y'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT)),
        'z': PIDController(**PID_GAINS['z'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT)),
    }

    # Benchmarking Variables
    benchmark_data = []
    start_wall_time = time.time()
    
    current_target_idx = 0
    state = "MOVING"
    steady_counter = 0
    wait_counter = 0
    MAX_STEPS = 50000 

    for step in range(MAX_STEPS):
        s_x = p.getJointState(robotId, 0)
        s_y = p.getJointState(robotId, 1)
        s_z = p.getJointState(robotId, 2)
        
        if current_target_idx < len(targets_queue):
            target_pos = targets_queue[current_target_idx]
        else:
            target_pos = None
            if state != "FINISHED": state = "WAITING"

        action = [0, 0, 0, 0]

        if state == "MOVING":
            t_joints = calculate_joint_targets(robotId, target_pos, sim.pipette_offset)
            vx = pids['x'].update(t_joints[0], s_x[0], dt=SIM_TIMESTEP)
            vy = pids['y'].update(t_joints[1], s_y[0], dt=SIM_TIMESTEP)
            vz = pids['z'].update(t_joints[2], s_z[0], dt=SIM_TIMESTEP)
            action = [-vx, -vy, vz, 0]
            
            curr_pos = convert_joint_to_pipette_position(robotId, [s_x[0], s_y[0], s_z[0]], sim.pipette_offset)
            dist = np.linalg.norm(np.array(target_pos) - np.array(curr_pos))
            velocities = [s_x[1], s_y[1], s_z[1]]
            
            if dist < POS_THRESHOLD and all(abs(v) < VEL_THRESHOLD for v in velocities):
                steady_counter += 1
            else:
                steady_counter = 0
            
            if steady_counter >= STEADY_FRAMES:
                # Log Benchmarking Data at the moment of drop
                elapsed = time.time() - start_wall_time
                error_xy = np.linalg.norm(np.array(target_pos[:2]) - np.array(curr_pos[:2]))
                
                benchmark_data.append({
                    'id': current_target_idx,
                    'time': elapsed,
                    'target_xy': target_pos[:2],
                    'actual_xy': np.array(curr_pos[:2]),
                    'error': error_xy
                })
                
                state = "DROPPING"
                steady_counter = 0

        elif state == "DROPPING":
            action = [0, 0, 0, 1]
            wait_counter = 0
            state = "WAITING"

        elif state == "WAITING":
            action = [0, 0, 0, 0]
            wait_counter += 1
            if current_target_idx < len(targets_queue) and wait_counter >= WAIT_STEPS:
                current_target_idx += 1
                pids['x'].reset(); pids['y'].reset(); pids['z'].reset()
                state = "MOVING"
            
            if current_target_idx >= len(targets_queue) and wait_counter >= WAIT_STEPS:
                state = "FINISHED"
                break

        sim.run([action], num_steps=1)

    sim.close()

    # ==========================================
    # 4. RESULTS AND VISUALIZATION
    # ==========================================
    print("\n" + "="*85)
    print(f"{'Target':<7} | {'Time(s)':<8} | {'Target XY (m)':<25} | {'Actual XY (m)':<25} | {'Error (mm)':<10}")
    print("-" * 85)
    
    errors = []
    times = []
    for d in benchmark_data:
        t_str = f"[{d['target_xy'][0]:.4f}, {d['target_xy'][1]:.4f}]"
        a_str = f"[{d['actual_xy'][0]:.4f}, {d['actual_xy'][1]:.4f}]"
        error_mm = d['error'] * 1000
        errors.append(error_mm)
        times.append(d['time'])
        print(f"{d['id']:<7} | {d['time']:<8.2f} | {t_str:<25} | {a_str:<25} | {error_mm:<10.4f}")

    print("-" * 85)
    print(f"Mean Error: {np.mean(errors):.4f} mm  |  Total Time: {time.time() - start_wall_time:.2f} s")
    print("="*85)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Error Plot
    ax1.bar([f"R{d['id']}" for d in benchmark_data], errors, color='skyblue', edgecolor='navy')
    ax1.set_title("Drop Accuracy (Euclidean Error)")
    ax1.set_ylabel("Error (mm)")
    ax1.set_xlabel("Root ID")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Path Visualization
    targets = np.array([d['target_xy'] for d in benchmark_data])
    actuals = np.array([d['actual_xy'] for d in benchmark_data])
    ax2.scatter(targets[:,0], targets[:,1], c='red', marker='x', label='Target')
    ax2.scatter(actuals[:,0], actuals[:,1], edgecolors='blue', facecolors='none', s=80, label='Actual Drop')
    for i in range(len(targets)):
        ax2.plot([targets[i,0], actuals[i,0]], [targets[i,1], actuals[i,1]], 'k--', alpha=0.5)
    ax2.set_title("Spatial Accuracy: Target vs Actual")
    ax2.set_xlabel("Robot X (m)")
    ax2.set_ylabel("Robot Y (m)")
    ax2.legend()
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()