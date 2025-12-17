import pybullet as p
import time
import itertools
from task10_pid_controller.pid import PIDController
from sim_class import Simulation
from pid_utils import *

# ==========================================
# 1. TUNING CONFIGURATION
# ==========================================

# Tuning Grids (The ranges to test)
kp_grid = [5.0, 15.0, 35.0, 100.0, 300.0] # Expanded range based on your other script
ki_grid = [0.0, 0.005, 0.01]
kd_grid = [0.1, 0.2, 0.5, 1.0]

# Simulation Constants
SPEED_LIMIT = 3.0       
FORCE_LIMIT = 600       
POS_THRESHOLD = 0.001   # 1mm
VEL_THRESHOLD = 0.01    # Rad/s
STEADY_FRAMES = 15      
TIMEOUT = 4.0           
SIM_TIMESTEP = 1.0 / 240.0 # Fixed Physics Step

# Workspace Limits (To define trajectories)
X_LIMITS = [-0.1870, 0.2530]
Y_LIMITS = [-0.1705, 0.2195]
Z_LIMITS = [0.1195, 0.2895]

final_results = {}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_1d_trajectory(axis_name):
    """
    Returns start_pos, end_pos, and axis_index.
    Keeps non-active axes in the exact middle of the workspace.
    """
    mid_x = sum(X_LIMITS)/2
    mid_y = sum(Y_LIMITS)/2
    mid_z = sum(Z_LIMITS)/2
    
    m = 0.02 # Margin from wall
    
    if axis_name == 'x':
        start = [X_LIMITS[0]+m, mid_y, mid_z]
        end   = [X_LIMITS[1]-m, mid_y, mid_z]
        idx = 0
    elif axis_name == 'y':
        start = [mid_x, Y_LIMITS[0]+m, mid_z]
        end   = [mid_x, Y_LIMITS[1]-m, mid_z]
        idx = 1
    elif axis_name == 'z':
        start = [mid_x, mid_y, Z_LIMITS[0]+m]
        end   = [mid_x, mid_y, Z_LIMITS[1]-m]
        idx = 2
        
    return start, end, idx

def calculate_joint_targets(robotId, target_pos, pipette_offset):
    base_pos = list(p.getBasePositionAndOrientation(robotId)[0])
    target_joint_x = base_pos[0] + pipette_offset[0] - target_pos[0]
    target_joint_y = base_pos[1] + pipette_offset[1] - target_pos[1]
    target_joint_z = target_pos[2] - base_pos[2] - pipette_offset[2]
    return [target_joint_x, target_joint_y, target_joint_z]

# ==========================================
# 3. CORE TEST FUNCTION (UPDATED LOGIC)
# ==========================================

def run_1d_test(robotId, axis_idx, start_joints, target_joints, target_world, kp, ki, kd):
    """
    Runs a test using Fixed Time Steps for PID stability.
    """
    
    # 1. Teleport to Start
    p.resetJointState(robotId, 0, start_joints[0], targetVelocity=0)
    p.resetJointState(robotId, 1, start_joints[1], targetVelocity=0)
    p.resetJointState(robotId, 2, start_joints[2], targetVelocity=0)
    
    # 2. Config Controllers
    # Active Axis gets the Grid Values, others get stiff holding values
    hold_kp, hold_ki, hold_kd = 50.0, 0.01, 0.5
    
    pids = []
    for i in range(3):
        if i == axis_idx:
            pids.append(PIDController(kp, ki, kd, (-SPEED_LIMIT, SPEED_LIMIT)))
        else:
            pids.append(PIDController(hold_kp, hold_ki, hold_kd, (-SPEED_LIMIT, SPEED_LIMIT)))
    
    pid_x, pid_y, pid_z = pids[0], pids[1], pids[2]
    
    # Reset PID memory
    pid_x.reset()
    pid_y.reset()
    pid_z.reset()
    
    start_time = time.time()
    steady_counter = 0
    success = False
    final_error = 100.0
    
    # 3. Control Loop (Fixed Step Logic)
    while (time.time() - start_time) < TIMEOUT:
        
        # Readings
        s_x = p.getJointState(robotId, 0)
        s_y = p.getJointState(robotId, 1)
        s_z = p.getJointState(robotId, 2)
        
        # PID Update (USING FIXED SIM_TIMESTEP)
        vx = pid_x.update(target_joints[0], s_x[0], dt=SIM_TIMESTEP)
        vy = pid_y.update(target_joints[1], s_y[0], dt=SIM_TIMESTEP)
        vz = pid_z.update(target_joints[2], s_z[0], dt=SIM_TIMESTEP)
        
        # Actuation
        action = [-vx, -vy, vz, 0]

        # Run simulation for 1 step
        sim.run([action], num_steps=1)
        
        # OPTIONAL: Speed Control
        # Remove this line for maximum speed (Turbo Mode)
        # Keep it to watch the tuning happen at 4x speed
        
        
        # --- ERROR CALCULATION ---
        # Forward Kinematics to find pipette tip
        base_pos = list(p.getBasePositionAndOrientation(robotId)[0])
        base_pos[0] -= s_x[0]
        base_pos[1] -= s_y[0]
        base_pos[2] += s_z[0]
        curr_world = [
            base_pos[0]+sim.pipette_offset[0],
            base_pos[1]+sim.pipette_offset[1],
            base_pos[2]+sim.pipette_offset[2]
        ]
        
        # Isolated Error
        error_1d = abs(target_world[axis_idx] - curr_world[axis_idx])
        final_error = error_1d
        
        # Check Stop Condition
        velocities = [s_x[1], s_y[1], s_z[1]]
        is_stopped = abs(velocities[axis_idx]) < VEL_THRESHOLD
        
        if error_1d < POS_THRESHOLD and is_stopped:
            steady_counter += 1
        else:
            steady_counter = 0
            
        if steady_counter >= STEADY_FRAMES:
            success = True
            break
            
    # --- SAFETY STOP ---
    # Stop motors so momentum doesn't carry over to the next test
    p.setJointMotorControlArray(robotId, [0,1,2], p.VELOCITY_CONTROL, targetVelocities=[0,0,0])
            
    return success, (time.time() - start_time), final_error


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    sim = Simulation(num_agents=1, render=True) 
    robotId = sim.robotIds[0]
    
    # Global Physics Setup
    p.setTimeStep(SIM_TIMESTEP)

    print("=== STARTING ISOLATED AXIS TUNING ===")
    print(f"Physics Step: {SIM_TIMESTEP:.5f}s")
    
    # Map for easy looping
    axes_map = {'x': 0, 'y': 1, 'z': 2}

    for axis_name, axis_idx in axes_map.items():
        print(f"\n" + "="*50)
        print(f"🔎 TUNING AXIS: {axis_name.upper()} (1D Movement Only)")
        print("="*50)
        
        # 1. Define Trajectory
        p1, p2, _ = get_1d_trajectory(axis_name)
        
        start_joints = calculate_joint_targets(robotId, p1, sim.pipette_offset)
        end_joints = calculate_joint_targets(robotId, p2, sim.pipette_offset)
        
        print(f"Move From: {['%.3f'%x for x in p1]}")
        print(f"Move To:   {['%.3f'%x for x in p2]}")
        print("-" * 75)
        print(f"{'Kp':<6} {'Ki':<6} {'Kd':<6} | {'Result':<10} | {'Time (s)':<10} | {'Error (mm)':<10}")
        print("-" * 75)
        
        axis_results = []
        
        # 2. Grid Search
        for kp, ki, kd in itertools.product(kp_grid, ki_grid, kd_grid):
            
            success, duration, error = run_1d_test(
                robotId, axis_idx, start_joints, end_joints, p2, kp, ki, kd
            )
            
            status = "SUCCESS" if success else "FAIL"
            # Color code output
            if success:
                print(f"{kp:<6} {ki:<6} {kd:<6} | {status:<10} | {duration:.4f}     | {error*1000:.4f}")
                axis_results.append({
                    'kp': kp, 'ki': ki, 'kd': kd, 
                    'time': duration, 'error': error
                })
            else:
                # Optional: Print fails sparsely or not at all to reduce clutter
                print(f"{kp:<6} {ki:<6} {kd:<6} | {status:<10} | {duration:.4f}     | {error*1000:.4f}")

                
        # 3. Analyze Results
        if axis_results:
            # Sort by time (fastest first)
            best = sorted(axis_results, key=lambda x: x['time'])[0]
            final_results[axis_name] = {'kp': best['kp'], 'ki': best['ki'], 'kd': best['kd']}
            print(f"\n🏆 BEST {axis_name.upper()}: {final_results[axis_name]} (Time: {best['time']:.3f}s)")
        else:
            print(f"\n❌ Failed to tune {axis_name.upper()}. Using defaults.")
            final_results[axis_name] = {'kp': 100.0, 'ki': 0.005, 'kd': 0.2}

    # ==========================================
    # 5. FINAL REPORT
    # ==========================================
    print("\n\n" + "#"*50)
    print("🏁 FINAL DECOUPLED CONFIGURATION")
    print("#"*50)
    print("PID_GAINS = {")
    print(f"    'x': {final_results.get('x', {})},")
    print(f"    'y': {final_results.get('y', {})},")
    print(f"    'z': {final_results.get('z', {})},")
    print("}")

    while True:
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)