import pybullet as p
import time
import random
from task10_pid_controller.pid import PIDController
from sim_class import Simulation

# ==========================================
# 1. USER CONFIGURATION (Your Tuned Gains)
# ==========================================
PID_GAINS = {
    'x': {'kp': 25.0, 'ki': 0.3, 'kd': 0.4},
    'y': {'kp': 25.0, 'ki': 0.3, 'kd': 0.4},
    'z': {'kp': 25.0, 'ki': 0.000, 'kd': 0.1},
}

# Simulation Constants
SPEED_LIMIT = 1.0       # Max motor speed (rad/s)
FORCE_LIMIT = 600       # Max motor force
POS_THRESHOLD = 0.001   # 1mm (Client Requirement)
VEL_THRESHOLD = 0.01    # Rad/s (Considered "stopped")
STEADY_FRAMES = 15      # Frames to hold position to confirm stop
TIME_BETWEEN_MOVES = 0.5 # Seconds to pause after reaching target
TIMEOUT = 5.0           # Max seconds to try reaching a point

# Workspace Limits (The Safe Box)
X_LIMITS = [-0.1870, 0.2530]
Y_LIMITS = [-0.1705, 0.2195]
Z_LIMITS = [0.1195, 0.2895]
MARGIN = 0.05 # 5cm safety buffer from walls



# ==========================================
# 3. HELPER FUNCTIONS
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

def get_random_point():
    return [
        random.uniform(X_LIMITS[0]+MARGIN, X_LIMITS[1]-MARGIN),
        random.uniform(Y_LIMITS[0]+MARGIN, Y_LIMITS[1]-MARGIN),
        random.uniform(Z_LIMITS[0]+MARGIN, Z_LIMITS[1]-MARGIN)
    ]

# ==========================================
# 4. MAIN SIMULATION LOOP
# ==========================================

if __name__ == "__main__":
    # Initialize Sim
    sim = Simulation(num_agents=1, render=True)
    robotId = sim.robotIds[0]

    # Initialize Controllers with YOUR SPECIFIC GAINS
    pid_x = PIDController(**PID_GAINS['x'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT))
    pid_y = PIDController(**PID_GAINS['y'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT))
    pid_z = PIDController(**PID_GAINS['z'], output_limits=(-SPEED_LIMIT, SPEED_LIMIT))

    print("\n=== STARTING RANDOM WALK WITH TUNED GAINS ===")
    print(f"X Config: {PID_GAINS['x']}")
    print(f"Y Config: {PID_GAINS['y']}")
    print(f"Z Config: {PID_GAINS['z']}")
    print("=" * 60)

    point_counter = 1
    SIM_TIMESTEP = 1.0 / 240.0
    p.setTimeStep(SIM_TIMESTEP)
    
    while True:
        # 1. Generate New Target (Random point in workspace)
        target_world = get_random_point()
        target_joints = calculate_joint_targets(robotId, target_world, sim.pipette_offset)
        
        print(f"\nTarget {point_counter}: {['%.3f'%x for x in target_world]}")
        
        # 2. Reset PID Memory (Crucial for clean step response)
        pid_x.reset()
        pid_y.reset()
        pid_z.reset()
        
        move_start_time = time.time()
        steady_counter = 0
        final_dist = 100.0
        
        # 3. Movement Loop
        while True:
            # -- SENSORS --
            s_x = p.getJointState(robotId, 0)
            s_y = p.getJointState(robotId, 1)
            s_z = p.getJointState(robotId, 2)
            
            # -- PID CALCULATION --
            vx = pid_x.update(target_joints[0], s_x[0], SIM_TIMESTEP)
            vy = pid_y.update(target_joints[1], s_y[0], SIM_TIMESTEP)
            vz = pid_z.update(target_joints[2], s_z[0], SIM_TIMESTEP)
            
            # -- ACTUATION using sim.run() --
            # Note: apply_actions negates x and y velocities, so we invert them here
            action = [-vx, -vy, vz, 0]
            sim.run([action], num_steps=1)
            
            # -- CHECK COMPLETION --
            curr_pos = convert_joint_to_pipette_position(robotId, [s_x[0], s_y[0], s_z[0]], sim.pipette_offset)
            dist = sum([(t - c)**2 for t, c in zip(target_world, curr_pos)]) ** 0.5
            
            velocities = [s_x[1], s_y[1], s_z[1]]
            is_stopped = all(abs(v) < VEL_THRESHOLD for v in velocities)
            
            # Check if we are inside the 1mm bubble AND stopped
            if dist < POS_THRESHOLD and is_stopped:
                steady_counter += 1
            else:
                steady_counter = 0
            
            # If we have been stable for enough frames, movement is done
            if steady_counter >= STEADY_FRAMES:
                duration = time.time() - move_start_time
                print(f"✅ Reached in {duration:.3f}s | Final Error: {dist*1000:.4f} mm")
                break
            
            # Timeout Check
            if (time.time() - move_start_time) > TIMEOUT:
                print(f"⚠️ Timeout! Moved on. | Final Error: {dist*1000:.4f} mm")
                break

        # 4. Wait briefly before generating the next point
        time.sleep(TIME_BETWEEN_MOVES)
        point_counter += 1