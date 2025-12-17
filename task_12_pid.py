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
    
    # Run Inference
    predicted_mask, preprocess_info = inference(image_path, model, patch_size, threshold=0.3)
    root_bottoms = get_bottom(predicted_mask, preprocess_info)

    # --- PHASE 2: COORDINATE TRANSFORMATION ---
    targets_queue = []
    pixel_points = [] # For visualization
    
    img_width = preprocess_info.crop_size
    plate_size_pixels = img_width 
    conversion_factor = PLATE_SIZE_MM / plate_size_pixels
    
    print(f"[INFO] Plate Scale: {plate_size_pixels}px = {PLATE_SIZE_MM}mm")
    print(f"[INFO] Conversion Factor: {conversion_factor:.4f} mm/px")
    
    print("\n[INFO] Targets found:")
    for k, v in root_bottoms.items():
        if v is not None:
            # v is (col, row) relative to top-left
            # We save it directly for plotting (Image X=Col, Image Y=Row)
            pixel_points.append(v)
            
            # --- ROBOT COORDINATE MAPPING ---
            # Unpacking v into y_pixel, x_pixel effectively performs the 90-degree rotation 
            # needed for the robot frame (Image Row -> Robot X, Image Col -> Robot Y)
            y_pixel, x_pixel = v 
            
            # 1. Pixel -> mm
            x_mm = x_pixel * conversion_factor
            y_mm = y_pixel * conversion_factor
            
            # 2. mm -> Robot Frame (Meters)
            dx_meter = x_mm / 1000.0
            dy_meter = y_mm / 1000.0
            
            # 3. Apply Plate Offset
            final_x = PLATE_POSITION_ROBOT[0] + dx_meter
            final_y = PLATE_POSITION_ROBOT[1] + dy_meter
            final_z = 0.2 
            
            final_pos = np.array([final_x, final_y, final_z], dtype=np.float32)
            targets_queue.append(final_pos)
            print(f"   Root {k}: Pixel({v[0]}, {v[1]}) -> Robot({final_pos[0]:.4f}, {final_pos[1]:.4f})")

    if not targets_queue:
        print("[WARN] No roots detected. Exiting.")
        exit()

    # --- PHASE 2.5: VISUALIZATION (ONCE) ---
    

#[Image of Cartesian coordinate system]

    print("\n[INFO] Displaying Targets. Close window to start simulation...")
    original_img = mpimg.imread(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(original_img[preprocess_info.crop_y: preprocess_info.crop_y+preprocess_info.crop_size,
                            preprocess_info.crop_x: preprocess_info.crop_x+preprocess_info.crop_size], cmap='gray')
    
    if pixel_points:
        # Zip creates lists of Xs and Ys from the (x,y) tuples
        xs, ys = zip(*pixel_points)
        plt.scatter(xs, ys, c='red', s=50, marker='x', label='Target Roots')
        plt.legend()
    
    plt.title("Detected Roots (Close this window to continue)")
    plt.show() # Code blocks here until you close the window
    
    # --- PHASE 3: SIMULATION LOOP ---
    print("\n[INFO] Starting Simulation...")
    
    robotId = sim.robotIds[0]
    p.setTimeStep(SIM_TIMESTEP)

    pids = {
        'x': PIDController(**PID_GAINS['x'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT)),
        'y': PIDController(**PID_GAINS['y'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT)),
        'z': PIDController(**PID_GAINS['z'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT)),
    }

    current_target_idx = 0
    state = "MOVING"
    steady_counter = 0
    wait_counter = 0
    MAX_STEPS = 100000 

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
                state = "DROPPING"
                steady_counter = 0

        elif state == "DROPPING":
            print(f"   [Step {step}] Dropping at Target {current_target_idx}")
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

        elif state == "FINISHED":
            if step % 240 == 0: 
                print("[INFO] All drops completed.")

        sim.run([action], num_steps=1)

    print("Simulation finished.")
    sim.close()