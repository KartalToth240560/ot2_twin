### Overview

This project implements a custom PID control system for the Opentrons OT-2 to achieve sub-millimeter positioning accuracy (< 1mm). The system uses three independent controllers (X, Y, Z) with a custom Inverse Kinematics solver to translate Cartesian coordinates into joint angles.

### Tuning Strategy: Decoupled 1D Grid Search

To optimize performance without the computational cost of a global search, I used a Decoupled 1D Grid Search:

Isolation: Restricted movement to one axis at a time.

Grid Search: Iterated through $K_p$, $K_i$, $K_d$ combinations for the active axis.

Selection: Selected gains yielding the fastest settling time with $< 1mm$ error.

Repeat: Process repeated sequentially for X, Y, and Z.

### Final Tuning Results

The decoupled search yielded the following optimal gains:
```
PID_GAINS = {
    'x': {'kp': 25.0, 'ki': 0.005, 'kd': 0.5},
    'y': {'kp': 25.0, 'ki': 0.005, 'kd': 0.5},
    'z': {'kp': 25.0, 'ki': 0.005, 'kd': 0.1},
}
```

### Performance Metrics

Testing involved random point-to-point movements across the workspace.

Positioning Error: ~0.3 mm (Requirement: < 1.0 mm)

Settling Time: ~0.5 seconds

Behavior: Fast rise time with minimal overshoot; critical damping achieves stability quickly.