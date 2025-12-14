import pybullet as p
import time
import random

# Constants

# Best PID by grid search
PID_GAINS = {
    'x': {'kp': 15.0, 'ki': 0.01, 'kd': 0.1},
    'y': {'kp': 15.0, 'ki': 0.01, 'kd': 0.1},
    'z': {'kp': 20.0, 'ki': 0.005, 'kd': 0.1},
}

# Simulation Constants
SPEED_LIMIT = 1.0       # Max motor speed (rad/s)
FORCE_LIMIT = 600       # Max motor force
WORKSPACE_THRESHOLD = 0.0001

POS_THRESHOLD = 0.001   # 1mm (Client Requirement)
VEL_THRESHOLD = 0.01    # Rad/s (Considered "stopped")
STEADY_FRAMES = 15      # Frames to hold position to confirm stop
TIME_BETWEEN_MOVES = 0.5 # Seconds to pause after reaching target
TIMEOUT = 5.0           # Max seconds to try reaching a point

# Workspace Limits (The Safe Box)
X_LIMITS = [-0.1870, 0.2530]
Y_LIMITS = [-0.1705, 0.2195]
Z_LIMITS = [0.1195, 0.2895]
MARGIN = 0.05 # 1cm safety buffer from walls


def calculate_joint_targets(robotId, target_pos, pipette_offset):
    base_pos = list(p.getBasePositionAndOrientation(robotId)[0])
    target_joint_x = base_pos[0] + pipette_offset[0] - target_pos[0]
    target_joint_y = base_pos[1] + pipette_offset[1] - target_pos[1]
    target_joint_z = target_pos[2] - base_pos[2] - pipette_offset[2]
    return [target_joint_x, target_joint_y, target_joint_z]

def convert_joint_to_pipette_position(robotId, joint_values, pipette_offset):
    """Convert joint positions to actual pipette X, Y, Z coordinates"""
    
    # Joint positions are just numbers that tell you how much each joint has moved.
    # For example: joint_0 = -0.15 means "joint 0 moved -0.15 meters"
    #
    # BUT these numbers don't tell you where the pipette tip actually is in 3D space.
    #
    # The pipette position is the ACTUAL X, Y, Z coordinate in world space where 
    # your pipette tip is located.
    
    # Set the joints to the specified values
    p.resetJointState(robotId, 0, joint_values[0])
    p.resetJointState(robotId, 1, joint_values[1])
    p.resetJointState(robotId, 2, joint_values[2])
    
    # Get robot base position
    robot_position = list(p.getBasePositionAndOrientation(robotId)[0])
    
    # Adjust for joint movements
    robot_position[0] -= joint_values[0]  # X joint moves in negative direction
    robot_position[1] -= joint_values[1]  # Y joint moves in negative direction
    robot_position[2] += joint_values[2]  # Z joint moves in positive direction
    
    # Add pipette offset
    pipette_pos = [
        robot_position[0] + pipette_offset[0],
        robot_position[1] + pipette_offset[1],
        robot_position[2] + pipette_offset[2]
    ]
    
    return pipette_pos