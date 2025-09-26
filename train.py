import time
import pygame
import numpy as np
import torch
import json
import os
import cv2
import argparse
from collections import deque
from torch.amp import autocast
from settings import *
from ppo_agent import PPOAgent
from utils import RunningMeanStd
from env import JumpKingEnv
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import FrameStackObservation, RecordEpisodeStatistics

os.environ['SDL_VIDEODRIVER'] = 'dummy' # headless mode
pygame.init()
pygame.display.set_mode((1, 1), pygame.NOFRAME) 

load_all_images()
from player import Player
Player.load_sprites()

class Hyperparameters:
    def __init__(self): 
        self.NUM_ENVS = 16 # Number of environments to sample
        self.LR = 5e-5 # Learning rate
        self.GAMMA = 0.99 # Discount factor
        self.GAE_LAMBDA = 0.95 # Lambda parameter for GAE
        self.N_EPOCHS = 4 # Number of training epochs per update
        self.EPS_CLIP = 0.2 # PPO clipping parameter
        self.ENTROPY_COEFF = 0.01 # Entropy regularization coefficient
        self.N_STEPS = 512 # Number of steps to collect from one environment per update
        self.MINI_BATCH_SIZE = 512 # Mini-batch size per update
        self.TOTAL_STEPS = 5e5 # Total sampling steps
        self.MAX_EPISODE_LENGTH = 1000 # Max steps in one episode
        self.TOTAL_UPDATES = int(self.TOTAL_STEPS / self.NUM_ENVS / self.N_STEPS)
        self.LOG_INTERVAL_UPDATES = 1
        self.SAVE_INTERVAL_UPDATES = 5
        self.CHECKPOINT_BASENAME = lambda level: f"training_checkpoint_level_{level + 1}.json"
        self.MODEL_SAVE_BASENAME = "ppo_model_level"

class JumpKingPPO_Sync:
    def __init__(self, hyperparams, target_level, base_model_level=None):
        
        # Initialization
        self.hp = hyperparams
        self.render_env = False # For debugging
        self.current_target_level = target_level - 1
        self.base_model_level = base_model_level

        print("Preloading level data...")
        from levelSetupFunction import MAP_LINES
        self.shared_levels = MAP_LINES
        print("Level data preloading complete.")
        
        # Environment creation functions
        env_fns = [self.make_env(rank=i, # Index of the environment
                                     initial_target_level=self.current_target_level,
                                     max_episode_steps=self.hp.MAX_EPISODE_LENGTH,
                                     render_human=(self.render_env and i == 0),
                                     shared_levels=self.shared_levels) 
                   for i in range(self.hp.NUM_ENVS)]

        # Synchronous vectorized environment
        self.vec_env = SyncVectorEnv(env_fns)
        
        # Observation space
        image_obs_shape = self.vec_env.single_observation_space.shape

        # Action space
        single_action_space = self.vec_env.single_action_space
        discrete_action_dims = [sp.n for sp in single_action_space.spaces]
    
        self.agent = PPOAgent(
            image_observation_shape=image_obs_shape, action_dims=discrete_action_dims, 
            n_steps=self.hp.N_STEPS, num_envs=self.hp.NUM_ENVS, 
            total_training_updates=self.hp.TOTAL_UPDATES, lr=self.hp.LR, gamma=self.hp.GAMMA, 
            n_epochs=self.hp.N_EPOCHS, eps_clip=self.hp.EPS_CLIP, entropy_coeff=self.hp.ENTROPY_COEFF, 
            gae_lambda=self.hp.GAE_LAMBDA, mini_batch_size=self.hp.MINI_BATCH_SIZE
        )
        
        self.reward_scaler = RunningMeanStd(shape=(self.hp.NUM_ENVS,)) # Reward scaling
        self.discounted_returns = np.zeros(self.hp.NUM_ENVS, dtype=np.float32) # Discounted returns
        self.total_env_steps_collected = 0
        self.global_update = 0
        self.completed_episode_rewards = deque(maxlen=100)
        self.completed_episode_lengths = deque(maxlen=100)
        self.best_height = 0
        self.training_log_data = []

        self.load_base_model()
        self.load_training_progress()
        self.update_env_target_level(self.current_target_level)


    # Function to create an environment instance
    def make_env(self, rank, initial_target_level, max_episode_steps, render_human=False, shared_levels=None):
        def _init():
            env = JumpKingEnv(initial_target_level=initial_target_level, render_mode="human" if render_human else None, max_episode_steps=max_episode_steps, shared_levels=shared_levels)
            env = FrameStackObservation(env, stack_size=FRAME_STACK_SIZE)
            env = RecordEpisodeStatistics(env)
            return env
        return _init
    
    # Set the training level for all environments
    def update_env_target_level(self, target_level_idx):
        if self.vec_env is None: return
        self.vec_env.call("set_player_target_level", target_level_idx)
    
    def load_base_model(self):
        if self.base_model_level is None or self.base_model_level <= 0: return

        model_filename = f"{self.hp.MODEL_SAVE_BASENAME}_{self.base_model_level}.pth"
        model_path = os.path.join(SAVE_PATH, model_filename)

        if os.path.exists(model_path):
            self.agent.load_model(model_path, set_to_training_mode=True)
        else: print(f"Failed to find the specified model: '{model_path}'.")

    def load_training_progress(self):
        checkpoint_path = os.path.join(CHECKPOINT_PATH, self.hp.CHECKPOINT_BASENAME(self.current_target_level))
        if not os.path.exists(checkpoint_path): return
        try:
            with open(checkpoint_path, 'r') as f: checkpoint_data = json.load(f)
            self.global_update = checkpoint_data.get('global_update', 0)
            if hasattr(self.agent, 'current_update_num'):
                self.agent.current_update_num = self.global_update
            self.total_env_steps_collected = checkpoint_data.get('total_env_steps_collected', 0)
            self.completed_episode_rewards = deque(checkpoint_data.get('completed_episode_rewards', []), maxlen=100)
            self.completed_episode_lengths = deque(checkpoint_data.get('completed_episode_lengths', []), maxlen=100)
            self.best_height = checkpoint_data.get('best_height', 0)
            self.training_log_data = checkpoint_data.get('training_log_data', [])
        except Exception as e: print(f"Failed to load checkpoint: {e}.")

    def save_checkpoint(self, is_level_completion_save=False):
        model_filename = f"{self.hp.MODEL_SAVE_BASENAME}_{self.current_target_level + 1}.pth"
        model_full_save_path = os.path.join(SAVE_PATH, model_filename)
        if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
        self.agent.save_model(model_full_save_path)
        
        checkpoint_data = {'model_filename': model_filename, 'current_target_level': int(self.current_target_level), 
                           'global_update': int(self.global_update), 'total_env_steps_collected': int(self.total_env_steps_collected), 
                           'completed_episode_rewards': [float(r) for r in self.completed_episode_rewards], 'completed_episode_lengths': [int(l) for l in self.completed_episode_lengths], 
                           'best_height': float(self.best_height), 'training_log_data': self.training_log_data, 
                           'is_level_completion_save': is_level_completion_save}
        
        checkpoint_path = os.path.join(CHECKPOINT_PATH, self.hp.CHECKPOINT_BASENAME(self.current_target_level))
        if not os.path.exists(CHECKPOINT_PATH): os.makedirs(CHECKPOINT_PATH)
        try:
            with open(checkpoint_path, 'w') as f: json.dump(checkpoint_data, f, indent=4)
            print(f"Checkpoint saved. Path: {checkpoint_path}")
        except Exception as e: print(f"Error saving checkpoint: {e}")
        
    def save_training_log(self):
        log_file_path = os.path.join(LOG_PATH, f"training_log_level_{self.current_target_level + 1}.json")
        if not os.path.exists(LOG_PATH): os.makedirs(LOG_PATH)
        try:
            with open(log_file_path, 'w') as f: json.dump(self.training_log_data, f, indent=4)
        except Exception as e: print(f"Error saving log: {e}")
    
    def run_training_loop(self):
        print(f"Training starts (Level: {self.current_target_level + 1})")
        start_time = time.time()
        self.update_env_target_level(self.current_target_level)
        
        current_image_observations, infos = self.vec_env.reset(options={'start_level': self.current_target_level})
        if 'global_height' in infos and infos['global_height'].size > 0:
            self.best_height = max(self.best_height, np.max(infos['global_height']))
        
        updates = 0
        while updates < self.hp.TOTAL_UPDATES:
            self.agent.clear_buffer() # Clear buffer, specific to on-policy algorithms
            
            for step_idx in range(self.hp.N_STEPS):
                with autocast(device_type="cuda", enabled=torch.cuda.is_available()): # Automatically select precision
                    discrete_actions_batch, values_ext_batch, log_probs_batch = self.agent.select_action(current_image_observations)
                
                obs_for_buffer = current_image_observations
                accumulated_rewards = np.zeros(self.hp.NUM_ENVS, dtype=np.float32)
                final_dones = np.zeros(self.hp.NUM_ENVS, dtype=bool) # Flag for whether an episode is done

                for frame_step_idx in range(ACTION_DECISION_INTERVAL):
                    action_to_vec_env = (discrete_actions_batch[:, 0].astype(np.int32), discrete_actions_batch[:, 1].astype(np.int32)) # Convert action format
                    new_image_observations, raw_rewards, terminateds, truncateds, infos = self.vec_env.step(action_to_vec_env) # Execute action
                    dones = np.logical_or(terminateds, truncateds)

                    '''
                    Moving Average for reward scaling, where recent rewards have higher weight.
                    Calculate the standard deviation of discounted returns for each environment, 
                    then scale the current reward by dividing it by the standard deviation.
                    This ensures that rewards remain stable and avoids training instability due to large reward variations.
                    '''
                    self.discounted_returns = self.hp.GAMMA * self.discounted_returns * (1 - dones) + raw_rewards 
                    self.reward_scaler.update(self.discounted_returns)
                    scaled_rewards = raw_rewards / (self.reward_scaler.std() + 1e-8)

                    if 'global_height' in infos: self.best_height = max(self.best_height, float(np.max(infos['global_height'])))
                    accumulated_rewards += scaled_rewards # Accumulated rewards after frame skip
                    final_dones = np.logical_or(final_dones, dones)
                    self.total_env_steps_collected += self.hp.NUM_ENVS

                    # Record episode information
                    if "episode" in infos and np.any(infos.get("_episode")):
                        ep_rewards = infos["episode"]["r"][infos["_episode"]]
                        ep_lengths = infos["episode"]["l"][infos["_episode"]]
                        for r, l in zip(ep_rewards, ep_lengths):
                            self.completed_episode_rewards.append(r)
                            self.completed_episode_lengths.append(l)

                    # Check the maximum height upon truncation
                    if "_final_info" in infos and np.any(infos.get("_final_info")):
                        for info in infos["final_info"][infos["_final_info"]]:
                            if info and 'global_height' in info and info['global_height'] > self.best_height:
                                self.best_height = info['global_height']

                    current_image_observations = new_image_observations
                    if np.all(dones): break
                
                self.agent.store_transitions_batch(
                    obs_for_buffer, discrete_actions_batch, 
                    log_probs_batch, accumulated_rewards, final_dones,
                    values_ext_batch
                )

            with torch.no_grad(), autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                # Bootstrap with the last value if the episode is not done
                _, last_values_ext_np, _ = self.agent.select_action(current_image_observations) 

            avg_actor_loss, avg_critic_loss, avg_entropy = self.agent.update(
                last_values_ext_np, final_dones
            )

            updates += 1
            self.global_update += 1
            
            # Log recording
            if self.global_update % self.hp.LOG_INTERVAL_UPDATES == 0:
                avg_rew = np.mean(self.completed_episode_rewards) if self.completed_episode_rewards else 0.0
                avg_len = np.mean(self.completed_episode_lengths) if self.completed_episode_lengths else 0.0
                end_time = time.time() - start_time
                height_to_print = float(self.best_height)
                print(f"Updates: {self.global_update} | Level: {self.current_target_level+1}")
                print(f"  Reward({len(self.completed_episode_rewards)}eps): {avg_rew:.2f} | Length: {avg_len:.1f} | Best_height: {height_to_print:.1f}")
                
                loss_str = f"Actor: {avg_actor_loss:.4f}, Critic: {avg_critic_loss:.4f}"
                print(f"  Loss -> {loss_str} | Entropy: {avg_entropy:.4f} | Time: {end_time:.1f}s\n")
                
                log_entry = {
                    'global_update': self.global_update, 'avg_reward': float(f"{avg_rew:.4f}"), 
                    'avg_length': float(f"{avg_len:.1f}"), 'best_height': float(f"{height_to_print:.1f}"), 
                    'avg_entropy': float(f"{avg_entropy:.4f}"), 'avg_actor_loss': float(f"{avg_actor_loss:.4f}"), 
                    'avg_critic_loss': float(f"{avg_critic_loss:.4f}")
                }
                self.training_log_data.append(log_entry)
                self.save_training_log()

            if self.global_update > 0 and self.global_update % self.hp.SAVE_INTERVAL_UPDATES == 0: 
                self.save_checkpoint()
            
        self.save_checkpoint()
        print(f"--- Training for target level {self.current_target_level + 1} completed ---")
        self.vec_env.close()
        pygame.quit()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Jump King agent with PPO and a custom CNN.")
    parser.add_argument("--target_level", type=int, default=5, help="The target level number to train for (starts from 1).")
    parser.add_argument("--base_model_level", type=int, default=None, help="Optional: The level from which to load a base model to start training.")
    
    args = parser.parse_args()
    
    hyperparameters = Hyperparameters()
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    if not os.path.exists(CHECKPOINT_PATH): os.makedirs(CHECKPOINT_PATH)
        
    trainer = JumpKingPPO_Sync( 
        hyperparams=hyperparameters, target_level=args.target_level, 
        base_model_level=args.base_model_level
    )
    trainer.run_training_loop()

