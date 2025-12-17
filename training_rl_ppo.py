# training_rl_ppo.py

import argparse
import os

# --- Suppress Tensorflow Warning --- 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import wandb
import gymnasium as gym


from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
# 1. Custom Checkpoint Callback that uploads to ClearML
class ClearMLCheckpointCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        # Call the parent class to save the model locally
        result = super()._on_step()
        
        # If a save happened in this step, upload it to ClearML
        if self.n_calls % self.save_freq == 0:
            # Reconstruct the path exactly as SB3 does
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            if os.path.exists(path):
                print(f"Uploading Checkpoint: {path}")
                # Use Task.current_task() to get the active task on the remote worker
                Task.current_task().upload_artifact(
                    name=f"checkpoint_{self.num_timesteps}", 
                    artifact_object=path
                )
        return result

# 2. Callback to upload the "Best Model" found by EvalCallback
class ClearMLBestModelCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        
    def _on_step(self) -> bool:
        # This is triggered ONLY when a new best model is found
        path = os.path.join(self.save_path, "best_model.zip")
        if os.path.exists(path):
            print(f"Uploading New Best Model: {path}")
            Task.current_task().upload_artifact(
                name="best_model", 
                artifact_object=path
            )
        return True

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

# Import your custom environment class
from ot2_twin.ot2_gym_wrapper import OT2Env

from clearml import Task

task = Task.init(project_name='Mentor Group - Uther/Group 1', # NB: Replace YourName with your own name
                    task_name='Experiment1')


# Force these specific package versions
task.set_packages([
    "numpy<2.0",                # <--- CRITICAL FIX: Downgrade to 1.x to fix binary incompatibility
    "gymnasium==1.2.2",
    "stable-baselines3==2.7.0",
    "wandb==0.23.1",
    "pybullet==3.2.5",
    "tensorboard==2.20.0",
    "pandas",
    "matplotlib",
    "clearml==2.0.2",
    "scikit-learn",
    "scipy",
    "shimmy>=0.2.1" 
])
#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue



# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--total_timesteps", type=int, default=1000)

args = parser.parse_args()


task.connect(args) 

# --- EXECUTE REMOTELY ---
# Now ClearML knows the args are 3,000,000 before it sends the task away
task.execute_remotely(queue_name="default")

# PROTECT THE ENTRY POINT (Required for Windows Multiprocessing)
if __name__ == "__main__":
    

    # --- ENVIRONMENT SETUP ---
    # make_vec_env automatically adds the 'Monitor' wrapper for you!
    # SubprocVecEnv puts each simulation in a separate process to prevent the crash.
    
    # 1. Training Environment
    env = make_vec_env(OT2Env, n_envs=1, vec_env_cls=SubprocVecEnv)

    # 2. Evaluation Environment
    eval_env = make_vec_env(OT2Env, n_envs=1, vec_env_cls=SubprocVecEnv)

    # --- WANDB SETUP ---
    os.environ['WANDB_API_KEY'] = '8e799aa123be181d9ce1b5bfd3e52bbc6da7a4b5'
    
    
    run = wandb.init(
        project="ot2_twin", 
        sync_tensorboard=True,
        monitor_gym=True 
    )

    # --- MODEL SETUP ---
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1, 
        learning_rate=args.learning_rate, 
        batch_size=args.batch_size, 
        n_steps=args.n_steps, 
        n_epochs=args.n_epochs, 
        tensorboard_log=f"runs/{run.id}"
    )

    # --- CALLBACK SETUP ---
    save_freq = int(args.total_timesteps / 10)
    
    # Define paths
    best_model_path = f"models/{run.id}/best_model"
    checkpoint_path = f"models/{run.id}/checkpoints"

    # 1. Stop Training Logic
    stop_train_callback = StopTrainingOnRewardThreshold(
        reward_threshold=-0.001, 
        verbose=1
    )

    # 2. ClearML Upload Logic for Best Model
    upload_best_callback = ClearMLBestModelCallback(save_path=best_model_path)

    # Combine them: When a new best model is found, check if we should stop AND upload the file
    callback_on_best = CallbackList([stop_train_callback, upload_best_callback])

    # 3. Eval Callback (Updated)
    eval_callback = EvalCallback(
        eval_env, 
        callback_on_new_best=callback_on_best,  # <--- Use the combined list here
        eval_freq=2048, 
        best_model_save_path=best_model_path, 
        log_path=f"runs/{run.id}", 
        deterministic=True, 
        render=False
    )

    # 4. Checkpoint Callback (Updated to custom class)
    checkpoint_callback = ClearMLCheckpointCallback(  # <--- Use Custom Class
        save_freq=save_freq,
        save_path=checkpoint_path,
        name_prefix="rl_model"
    )

    # 5. Wandb Callback (Remains the same)
    wandb_callback = WandbCallback(
        model_save_freq=save_freq,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    callbacks_list = [eval_callback, checkpoint_callback, wandb_callback]

    # --- START TRAINING ---
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps, 
            callback=callbacks_list, 
            progress_bar=True, 
            reset_num_timesteps=False,
            tb_log_name=f"runs/{run.id}"
        )
        
        # Save final model
        model.save(f"models/{run.id}/final_model")
        task.upload_artifact("model", artifact_object=f"{run.id}.zip")
        print("Training finished successfully.")
        
    finally:
        # Cleanup
        env.close()
        eval_env.close()
        run.finish()