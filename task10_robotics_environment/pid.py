class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits # tuple (min, max) for clamping velocity
        
        self.prev_error = 0
        self.integral = 0
        
        
    def update(self, target, measured, dt):
        # 1. Calculate time delta (dt)
        
            
        
        if dt <= 0: return 0.0
        
        # 2. Calculate Error
        error = target - measured
        
        # 3. Proportional Term
        P = self.kp * error
        
        # 4. Integral Term
        self.integral += error * dt
        # Optional: Clamp integral to prevent windup if needed
        I = self.ki * self.integral
        
        # 5. Derivative Term
        derivative = (error - self.prev_error) / dt
        D = self.kd * derivative
        
        # 6. Calculate Output
        output = P + I + D
        
        # 7. Clamp Output (Limit max motor velocity)
        min_out, max_out = self.output_limits
        if min_out is not None and output < min_out:
            output = min_out
        if max_out is not None and output > max_out:
            output = max_out
            
        # 8. Save state for next step
        self.prev_error = error
        
        
        return output
    
    def reset(self):
        """Reset memory of the PID (integral and previous error)"""
        self.prev_error = 0
        self.integral = 0
        self.last_time = None