import pybullet as p
import time
import matplotlib.pyplot as plt
from task10_robotics_environment.pid import PIDController
from sim_class import Simulation
from pid_utils import * # Assuming your helper functions are here, or defined below

# ==========================================
# 1. CONFIGURATION
# ==========================================
PID_GAINS = {
    'x': {'kp': 25.0, 'ki': 0.005, 'kd': 0.5},
    'y': {'kp': 25.0, 'ki': 0.005, 'kd': 0.5},
    'z': {'kp': 25.0, 'ki': 0.005, 'kd': 0.1},
}

# Simulation Constants
SPEED_LIMIT = 3.0       
FORCE_LIMIT = 600       
POS_THRESHOLD = 0.001   # 1mm
VEL_THRESHOLD = 0.01    # Rad/s
STEADY_FRAMES = 15      
TIMEOUT = 4.0           
SIM_TIMESTEP = 1.0 / 240.0 # Fixed Physics Step

# ==========================================
# 2. HELPER FUNCTIONS (If not in utils)
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
# 3. TELEMETRY RECORDING & PLOTTING
# ==========================================
def run_telemetry_test():
    sim = Simulation(num_agents=1, render=True) 
    robotId = sim.robotIds[0]
    
    # Set Global Physics Time
    p.setTimeStep(SIM_TIMESTEP)

    # Initialize Controllers
    pid_x = PIDController(**PID_GAINS['x'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT))
    pid_y = PIDController(**PID_GAINS['y'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT))
    pid_z = PIDController(**PID_GAINS['z'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT))

    # --- DEFINE TEST POINTS (Step Input) ---
    p1 = [-0.150, -0.150, 0.250] # Start
    p2 = [ 0.200,  0.150, 0.170] # End
    
    # 1. Teleport to Start (P1)
    start_joints = calculate_joint_targets(robotId, p1, sim.pipette_offset)
    p.resetJointState(robotId, 0, start_joints[0], targetVelocity=0)
    p.resetJointState(robotId, 1, start_joints[1], targetVelocity=0)
    p.resetJointState(robotId, 2, start_joints[2], targetVelocity=0)
    
    # 2. Target End (P2)
    target_joints = calculate_joint_targets(robotId, p2, sim.pipette_offset)
    
    print("Collecting Telemetry Data...")

    # --- DATA STORAGE ---
    history = {
        'time': [],
        'target_x': [], 'actual_x': [],
        'target_y': [], 'actual_y': [],
        'target_z': [], 'actual_z': []
    }
    
    # We track "Simulation Time", not "Wall Clock Time"
    current_sim_time = 0.0 
    steady_counter = 0

    # 3. Control Loop
    while current_sim_time < TIMEOUT:
        
        # Readings
        s_x = p.getJointState(robotId, 0)
        s_y = p.getJointState(robotId, 1)
        s_z = p.getJointState(robotId, 2)
        
        # Calculate Real-World Position (Actual)
        curr_pos = convert_joint_to_pipette_position(robotId, [s_x[0], s_y[0], s_z[0]], sim.pipette_offset)
        
        # --- RECORD HISTORY ---
        history['time'].append(current_sim_time)
        history['target_x'].append(p2[0])
        history['actual_x'].append(curr_pos[0])
        history['target_y'].append(p2[1])
        history['actual_y'].append(curr_pos[1])
        history['target_z'].append(p2[2])
        history['actual_z'].append(curr_pos[2])
        # -----------------------

        # PID Update (USING FIXED DT)
        vx = pid_x.update(target_joints[0], s_x[0], dt=SIM_TIMESTEP)
        vy = pid_y.update(target_joints[1], s_y[0], dt=SIM_TIMESTEP)
        vz = pid_z.update(target_joints[2], s_z[0], dt=SIM_TIMESTEP)
        
        # Actuation
        action = [-vx, -vy, vz, 0]
        sim.run([action], num_steps=1)
        # Physics Step
        
        # Update Time
        current_sim_time += SIM_TIMESTEP
        
        # Optional: Visual Delay (Does not affect data)
        # time.sleep(SIM_TIMESTEP) 
        
        # Check Stop Condition
        dist = sum([(t - c)**2 for t, c in zip(p2, curr_pos)]) ** 0.5
        velocities = [s_x[1], s_y[1], s_z[1]]
        is_stopped = all(abs(v) < VEL_THRESHOLD for v in velocities)
        
        if dist < POS_THRESHOLD and is_stopped:
            steady_counter += 1
        else:
            steady_counter = 0
            
        if steady_counter >= STEADY_FRAMES:
            print(f"✅ Target reached in {current_sim_time:.3f}s (Sim Time)")
            break

    # --- PLOTTING ---
    plot_results(history)

def plot_results(h):
    # Set up the figure with 3 subplots (rows or columns)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"PID Telemetry (Step Response)", fontsize=16)

    # --- X AXIS PLOT ---
    axs[0].plot(h['time'], h['target_x'], 'b-', label='Target', linewidth=2)
    axs[0].plot(h['time'], h['actual_x'], 'r-', label='Actual', linewidth=2)
    axs[0].set_title(f"X Axis Response (Kp={PID_GAINS['x']['kp']})")
    axs[0].set_xlabel("Sim Time (s)")
    axs[0].set_ylabel("Position (m)")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()

    # --- Y AXIS PLOT ---
    axs[1].plot(h['time'], h['target_y'], 'b-', label='Target', linewidth=2)
    axs[1].plot(h['time'], h['actual_y'], 'r-', label='Actual', linewidth=2)
    axs[1].set_title(f"Y Axis Response (Kp={PID_GAINS['y']['kp']})")
    axs[1].set_xlabel("Sim Time (s)")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()

    # --- Z AXIS PLOT ---
    axs[2].plot(h['time'], h['target_z'], 'b-', label='Target', linewidth=2)
    axs[2].plot(h['time'], h['actual_z'], 'r-', label='Actual', linewidth=2)
    axs[2].set_title(f"Z Axis Response (Kp={PID_GAINS['z']['kp']})")
    axs[2].set_xlabel("Sim Time (s)")
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_telemetry_test()