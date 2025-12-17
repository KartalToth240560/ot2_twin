# Task 09 - Robotics Environment

Finding workspace limits for a 3 DOF linear robot arm in PyBullet simulation.

![Finding minimums and maximums along axis'](task_9.gif)

## Environment Setup

### Prerequisites
- Python 3.11.14
- Conda package manager

### Installation

1. Create and activate a new conda environment:
```bash
conda create -n robotics_env python=3.11.14
conda activate robotics_env
```

2. Install PyBullet from conda-forge:
```bash
conda install -c conda-forge pybullet=3.25
```

3. Install required Python packages:
```bash
pip install numpy==2.2.6
pip install pandas==2.2.3
pip install matplotlib==3.10.7
```

### Project Structure
```
task09_robotics_environment/
├── README.md
├── task_9.py              # Main script for finding workspace limits
├── sim_class.py           # Simulation interface class
└── task_9.gif             # Demonstration visualization
```

**Note:** Ensure `sim_class.py` is in the same directory as `task_9.py` before running.

## Pipette Workspace Limits

The physical limits of the pipette tip in 3D space:
- **X:** [-0.1870, 0.2530] meters
- **Y:** [-0.1705, 0.2195] meters  
- **Z:** [0.1195, 0.2895] meters

**Total workspace dimensions:**
- X range: 0.44 meters (440mm)
- Y range: 0.39 meters (390mm)
- Z range: 0.17 meters (170mm)

## Method

The method for finding the limits of workspace were moving along each axis of XYZ with the linear joints of the robot, until the position of the robot position did not change further.

**Key Parameters:**
- Movement velocity: 0.5 m/s
- Motor force: 500N
- Position change threshold: 0.0001 meters

## Running

### Basic Execution
```bash
cd ot2_twin
python task_9.py
```

### Expected Output
The script will:
1. Initialize the PyBullet simulation environment
2. Load the robot model
3. Systematically test each axis (X, Y, Z) for minimum and maximum limits
4. Display real-time progress in the console
5. Output final workspace boundaries

### Execution Time
Approximately 2-3 minutes depending on system performance.

## Key Findings on Robot Behavior and Limitations

### Force Parameter Sensitivity
- **Too High Force (>1000N)**: The robot begins to jitter and oscillate when reaching joint limits. The motor fights against the physical constraints, causing unstable behavior.
- **Optimal Force**: 500N provides stable movement for X and Y axes. Z-axis may require higher force (up to 2000N) to overcome gravity.
- **Too Low Force**: Insufficient force causes the robot to move slowly or fail to reach actual joint limits.

### Threshold Parameter Issues
- **Too Small Threshold (<0.00001)**: Can cause the algorithm to get stuck, particularly on the Z-axis. The position detection becomes too sensitive and may trigger false "stopped" detections.
- **Optimal Threshold**: 0.0001 meters provides reliable detection of when the joint has reached its physical limit.
- **Too Large Threshold**: May report limits before the joint actually reaches its physical boundary.

### Z-Axis (Vertical) Specific Challenges
- **Gravity Effect**: The Z-axis is affected by gravity, making upward movement more difficult than downward movement.
- **Sticking Issue**: After moving down to the minimum limit, the joint can get stuck and jitter when trying to move back up.
- **Solution**: Use higher force values for Z-axis movements, or test maximum limit first (moving up) before minimum limit (moving down).

### Coordinate System Notes
- **Joint Movement Directions**:
  - Joint 0 (X): Negative joint values move in negative X direction
  - Joint 1 (Y): Negative joint values move in negative Y direction
  - Joint 2 (Z): Positive joint values move in positive Z direction
- **Pipette Offset**: The pipette tip is offset from the robot base by [0.073, 0.0895, 0.0895] meters. This offset must be added to calculate actual workspace coordinates.

### Simulation Timestep
- Physics simulation runs at 240 Hz (1/240 seconds per step)
- Movement testing uses 10 simulation steps per iteration
- Allow 0.5 seconds settling time after reaching limits to stabilize readings

## Algorithm Overview

The workspace limit finding algorithm:
1. Moves each joint in negative direction until movement stops (hits minimum limit)
2. Moves each joint in positive direction until movement stops (hits maximum limit)
3. Converts joint positions to real-world pipette coordinates using robot base position and pipette offset
4. Reports the actual 3D workspace boundaries

## Limitations and Considerations
- **Physical Simulation**: Results are based on PyBullet physics simulation and may differ slightly from real hardware
- **Collision Detection**: The algorithm detects physical joint limits, not collision with workspace objects
- **Single Robot Testing**: Workspace is determined for a single robot in isolation
- **Static Environment**: Does not account for dynamic obstacles or multi-robot interactions