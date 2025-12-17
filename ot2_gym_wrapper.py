import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


X_LIMITS = [-0.1870, 0.2530]
Y_LIMITS = [-0.1705, 0.2195]
Z_LIMITS = [0.1195, 0.2895]
MARGIN = 0.01 # 1cm safety buffer from walls
VEL_THRESHOLD = 0.01 # rad/sec
DIST_THRESHOLD = 0.001 # 1mm


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(np.array([-1, -1, -1], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32), (3,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

        # keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
        self.goal_position = [
            np.random.uniform(X_LIMITS[0]+MARGIN, X_LIMITS[1]-MARGIN),
            np.random.uniform(Y_LIMITS[0]+MARGIN, Y_LIMITS[1]-MARGIN),
            np.random.uniform(Z_LIMITS[0]+MARGIN, Z_LIMITS[1]-MARGIN)
        ]
        
        # Call the environment reset function
        sim_states = self.sim.reset(num_agents=1)
        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        observation = self.get_observation(sim_states)[:6]

        # Reset the number of steps
        self.steps = 0

        info = {}
        
        return observation, info
    def get_observation(self, sim_states):
        """
        Extract pipette position and joint velocities from the sim state.
        Returns numpy array:
        [px, py, pz, gx, gy, gz, vx, vy, vz]
        """
        # The simulation returns a dict: { "robotId_<id>": {...} }
        robot_key = next(iter(sim_states.keys()))
        robot_data = sim_states[robot_key]

        # pipette position
        pipette_x, pipette_y, pipette_z = robot_data["pipette_position"]

        # velocities from joints 0,1,2
        s_x = robot_data["joint_states"]["joint_0"]     # joint 0 → X axis
        s_y = robot_data["joint_states"]["joint_1"]     # joint 1 → Y axis
        s_z = robot_data["joint_states"]["joint_2"]     # joint 2 → Z axis
        
        vel_x = s_x["velocity"]
        vel_y = s_y["velocity"]
        vel_z = s_z["velocity"]

        # Build final observation vector
        obs = np.array([
            pipette_x, pipette_y, pipette_z,
            self.goal_position[0], self.goal_position[1], self.goal_position[2],
            vel_x, vel_y, vel_z
        ], dtype=np.float32)

        return obs
    def step(self, action):
        # Execute one time step within the environment
        # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = [action[0], action[1], action[2], 0]

        # Call the environment step function
        self.sim.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.
        g_obs = self.get_observation(self.sim.get_states())
        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        observation = g_obs[:6]

        # Calculate the reward, this is something that you will need to experiment with to get the best results
        # Euclidean distance between current position and goal
        dist = np.linalg.norm(observation[0:3] - observation[3:6])

        # Dense shaping reward
        reward = -dist

        # Sparse success bonus
        GOAL_THRESHOLD = 0.001  # 1 mm
        SUCCESS_BONUS = 10.0

        if dist < GOAL_THRESHOLD:
            reward += SUCCESS_BONUS
        
        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
        vel_x, vel_y, vel_z = g_obs[6:]
        
        
        is_stopped = all(abs(v) < VEL_THRESHOLD for v in [vel_x, vel_y, vel_z])

        if is_stopped:
            terminated = bool(dist < DIST_THRESHOLD)
        else:
            terminated = False
        truncated = self.steps >= self.max_steps
        

        info = {} # we don't need to return any additional information

        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()