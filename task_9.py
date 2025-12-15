import time
from pid_utils import *
from sim_class import Simulation


def find_joint_physical_limits(sim, robotId, joint_idx):
    """Move each joint until it physically cannot move anymore using sim.run()"""
    
    print(f"\nFinding limits for joint {joint_idx}...")
    
    # Create actions array for single robot: [x_vel, y_vel, z_vel, drop]
    # We'll set velocity for the target joint and 0 for others
    
    # Find minimum limit
    min_limit = None
    for step in range(1000):
        current_pos = p.getJointState(robotId, joint_idx)[0]
        
        # Create action with negative velocity for this joint
        action = [0.0, 0.0, 0.0, 0]
        if joint_idx == 0:
            action[0] = SPEED_LIMIT  # Note: negative velocity in apply_actions
        elif joint_idx == 1:
            action[1] = SPEED_LIMIT  # Note: negative velocity in apply_actions
        elif joint_idx == 2:
            action[2] = -SPEED_LIMIT  # Note: positive velocity in apply_actions
        
        # Run simulation for 10 steps
        sim.run([action], num_steps=10)
        
        new_pos = p.getJointState(robotId, joint_idx)[0]
        
        if abs(new_pos - current_pos) < WORKSPACE_THRESHOLD:
            min_limit = new_pos
            time.sleep(0.5)
            break
    
    # Find maximum limit
    max_limit = None
    for step in range(1000):
        current_pos = p.getJointState(robotId, joint_idx)[0]
        
        # Create action with positive velocity for this joint
        action = [0.0, 0.0, 0.0, 0]
        if joint_idx == 0:
            action[0] = -SPEED_LIMIT  # Note: negative velocity in apply_actions
        elif joint_idx == 1:
            action[1] = -SPEED_LIMIT  # Note: negative velocity in apply_actions
        elif joint_idx == 2:
            action[2] = SPEED_LIMIT  # Note: positive velocity in apply_actions
        
        # Run simulation for 10 steps
        sim.run([action], num_steps=10)
        
        new_pos = p.getJointState(robotId, joint_idx)[0]
        
        if abs(new_pos - current_pos) < WORKSPACE_THRESHOLD:
            max_limit = new_pos
            break
    
    return (min_limit, max_limit)


# Usage:
sim = Simulation(num_agents=1, render=True)
robotId = sim.robotIds[0]
time.sleep(1)

# Find joint limits (min, max)
joint_0_limits = find_joint_physical_limits(sim, robotId, 0)  # X axis
joint_1_limits = find_joint_physical_limits(sim, robotId, 1)  # Y axis
joint_2_limits = find_joint_physical_limits(sim, robotId, 2)  # Z axis

# Convert to actual pipette positions
positions_1 = convert_joint_to_pipette_position(
    robotId, 
    [joint_0_limits[0], joint_1_limits[0], joint_2_limits[0]], 
    sim.pipette_offset
)

positions_2 = convert_joint_to_pipette_position(
    robotId, 
    [joint_0_limits[1], joint_1_limits[1], joint_2_limits[1]], 
    sim.pipette_offset
)

min_x = min(positions_1[0], positions_2[0])
max_x = max(positions_1[0], positions_2[0])

min_y = min(positions_1[1], positions_2[1])
max_y = max(positions_1[1], positions_2[1])

min_z = min(positions_1[2], positions_2[2])
max_z = max(positions_1[2], positions_2[2])

print("\n=== PIPETTE WORKSPACE LIMITS ===")
print(f"X: [{min_x:.4f}, {max_x:.4f}]")
print(f"Y: [{min_y:.4f}, {max_y:.4f}]")
print(f"Z: [{min_z:.4f}, {max_z:.4f}]")

time.sleep(1)