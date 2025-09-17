import time
import pygame
import numpy as np
import torch
import json
import os
import argparse
from collections import deque
from torch.amp import autocast
from settings import *
from ppo_agent import PPOAgent
from utils import RunningMeanStd
from env import JumpKingEnv
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import FrameStackObservation, RecordEpisodeStatistics

os.environ['SDL_VIDEODRIVER'] = 'dummy' # 無頭模式
pygame.init()
pygame.display.set_mode((1, 1), pygame.NOFRAME) 

load_all_images()
from player import Player
Player.load_sprites()

class Hyperparameters:
    def __init__(self):
        self.NUM_ENVS = 16
        self.LR = 5e-5 
        self.GAMMA = 0.99 
        self.GAE_LAMBDA = 0.95
        self.N_EPOCHS = 4
        self.EPS_CLIP = 0.2
        self.ENTROPY_COEFF = 0.01
        self.N_STEPS = 512
        self.MINI_BATCH_SIZE = 512
        self.TOTAL_STEPS = 5e5
        self.MAX_EPISODE_STEPS_ENV = 900
        self.TOTAL_UPDATES = int(self.TOTAL_STEPS / self.NUM_ENVS / self.N_STEPS)
        self.LOG_INTERVAL_UPDATES = 1
        self.SAVE_INTERVAL_UPDATES = 5
        self.CHECKPOINT_BASENAME = lambda level: f"training_checkpoint_level_{level + 1}.json"
        self.MODEL_SAVE_BASENAME = "ppo_model_level"

class JumpKingPPO_Sync:
    def __init__(self, hyperparams, user_specified_target_level, base_model_level=None):
        
        # 初始化
        self.hp = hyperparams
        self.render_env = False # 偵錯用
        self.current_training_target_level = user_specified_target_level - 1
        self.base_model_level = base_model_level

        print("正在預先載入關卡資料...")
        from levelSetupFunction import MAP_LINES
        self.shared_levels = MAP_LINES
        print("關卡資料預載入完成。")
        
        # 環境函式
        env_fns = [self._make_env(rank=i, # 環境的索引
                                     initial_target_level=self.current_training_target_level,
                                     max_episode_steps=self.hp.MAX_EPISODE_STEPS_ENV,
                                     render_human=(self.render_env and i == 0),
                                     shared_levels=self.shared_levels) 
                   for i in range(self.hp.NUM_ENVS)]

        # 同步向量化環境
        self.vec_env = SyncVectorEnv(env_fns)
        
        # 觀測空間
        image_obs_shape = self.vec_env.single_observation_space.shape

        # 動作空間
        single_action_space = self.vec_env.single_action_space
        discrete_action_dims = [sp.n for sp in single_action_space.spaces]
    
        self.agent = PPOAgent(
            image_observation_shape=image_obs_shape, action_dims=discrete_action_dims, 
            n_steps=self.hp.N_STEPS, num_envs=self.hp.NUM_ENVS, 
            total_training_updates=self.hp.TOTAL_UPDATES, lr=self.hp.LR, gamma=self.hp.GAMMA, 
            n_epochs=self.hp.N_EPOCHS, eps_clip=self.hp.EPS_CLIP, entropy_coeff=self.hp.ENTROPY_COEFF, 
            gae_lambda=self.hp.GAE_LAMBDA, mini_batch_size=self.hp.MINI_BATCH_SIZE
        )
        
        self.reward_scaler = RunningMeanStd(shape=(self.hp.NUM_ENVS,)) # reward scaling
        self.discounted_returns = np.zeros(self.hp.NUM_ENVS, dtype=np.float32) # 折扣回報
        self.total_env_steps_collected = 0
        self.global_update = 0
        self.completed_episode_rewards = deque(maxlen=100)
        self.completed_episode_lengths = deque(maxlen=100)
        self.best_height_reached_overall = 0
        self.training_log_data = []

        self._load_base_model()
        self._load_training_progress()
        self._update_env_target_level(self.current_training_target_level)


    # 生成環境的函式
    def _make_env(self, rank, initial_target_level, max_episode_steps, render_human=False, shared_levels=None):
        def _init():
            env = JumpKingEnv(initial_target_level=initial_target_level, render_mode="human" if render_human else None, max_episode_steps=max_episode_steps, shared_levels=shared_levels)
            env = FrameStackObservation(env, stack_size=FRAME_STACK_SIZE)
            env = RecordEpisodeStatistics(env)
            return env
        return _init
    
    # 設定訓練關卡
    def _update_env_target_level(self, target_level_idx):
        if self.vec_env is None: return
        self.vec_env.call("set_player_target_level", target_level_idx)
    
    def _load_base_model(self):
        if self.base_model_level is None or self.base_model_level <= 0: return

        model_filename = f"{self.hp.MODEL_SAVE_BASENAME}_{self.base_model_level}.pth"
        model_to_load_path = os.path.join(SAVE_PATH, model_filename)

        if os.path.exists(model_to_load_path):
            self.agent.load_model(model_to_load_path, set_to_training_mode=True)
        else: print(f"找不到指定的模型 '{model_to_load_path}'。")

    def _load_training_progress(self):
        checkpoint_file_path = os.path.join(CHECKPOINT_PATH, self.hp.CHECKPOINT_BASENAME(self.current_training_target_level))
        if not os.path.exists(checkpoint_file_path): return
        try:
            with open(checkpoint_file_path, 'r') as f: checkpoint_data = json.load(f)
            self.global_update = checkpoint_data.get('global_update', 0)
            if hasattr(self.agent, 'current_update_num'):
                self.agent.current_update_num = self.global_update
            self.total_env_steps_collected = checkpoint_data.get('total_env_steps_collected', 0)
            self.completed_episode_rewards = deque(checkpoint_data.get('completed_episode_rewards', []), maxlen=100)
            self.completed_episode_lengths = deque(checkpoint_data.get('completed_episode_lengths', []), maxlen=100)
            self.best_height_reached_overall = checkpoint_data.get('best_height_reached_overall', 0)
            self.training_log_data = checkpoint_data.get('training_log_data', [])
        except Exception as e: print(f"無法加載檢查點 '{checkpoint_file_path}': {e}。")

    def save_checkpoint(self, is_level_completion_save=False):
        model_filename = f"{self.hp.MODEL_SAVE_BASENAME}_{self.current_training_target_level + 1}.pth"
        model_full_save_path = os.path.join(SAVE_PATH, model_filename)
        if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
        self.agent.save_model(model_full_save_path)
        
        checkpoint_data = {'model_filename': model_filename, 'current_training_target_level': int(self.current_training_target_level), 
                           'global_update': int(self.global_update), 'total_env_steps_collected': int(self.total_env_steps_collected), 
                           'completed_episode_rewards': [float(r) for r in self.completed_episode_rewards], 'completed_episode_lengths': [int(l) for l in self.completed_episode_lengths], 
                           'best_height_reached_overall': float(self.best_height_reached_overall), 'training_log_data': self.training_log_data, 
                           'is_level_completion_save': is_level_completion_save}
        
        checkpoint_file_path = os.path.join(CHECKPOINT_PATH, self.hp.CHECKPOINT_BASENAME(self.current_training_target_level))
        if not os.path.exists(CHECKPOINT_PATH): os.makedirs(CHECKPOINT_PATH)
        try:
            with open(checkpoint_file_path, 'w') as f: json.dump(checkpoint_data, f, indent=4)
            print(f"檢查點已保存到 {checkpoint_file_path} (包含模型 {model_filename})")
        except Exception as e: print(f"無法保存檢查點 '{checkpoint_file_path}': {e}")
        
    def save_training_log(self):
        log_file_path = os.path.join(LOG_PATH, f"training_log_level_{self.current_training_target_level + 1}.json")
        if not os.path.exists(LOG_PATH): os.makedirs(LOG_PATH)
        try:
            with open(log_file_path, 'w') as f: json.dump(self.training_log_data, f, indent=4)
        except Exception as e: print(f"無法將訓練日誌保存到 '{log_file_path}': {e}")
    
    # 訓練主循環
    def run_training_loop(self):
        print(f"開始 PPO 訓練 (目標關卡: {self.current_training_target_level + 1})")
        start_time = time.time()
        self._update_env_target_level(self.current_training_target_level)
        
        # 重製環境
        current_image_observations, infos = self.vec_env.reset(options={'start_level': self.current_training_target_level})
        if 'global_height' in infos and infos['global_height'].size > 0:
            self.best_height_reached_overall = max(self.best_height_reached_overall, np.max(infos['global_height']))
        
        updates = 0
        while updates < self.hp.TOTAL_UPDATES:
            self.agent.clear_buffer() # On-policy 特有的 clear buffer
            
            for step_idx in range(self.hp.N_STEPS):
                with autocast(device_type="cuda", enabled=torch.cuda.is_available()): # 自動選擇精度
                    discrete_actions_batch, values_ext_batch, log_probs_batch = self.agent.select_action(current_image_observations)
                
                obs_for_buffer = current_image_observations
                accumulated_rewards = np.zeros(self.hp.NUM_ENVS, dtype=np.float32)
                final_dones = np.zeros(self.hp.NUM_ENVS, dtype=bool) # 標記是否 done

                for frame_step_idx in range(ACTION_DECISION_INTERVAL):
                    action_to_vec_env = (discrete_actions_batch[:, 0].astype(np.int32), discrete_actions_batch[:, 1].astype(np.int32)) # 轉換動作格式
                    new_image_observations, raw_rewards, terminateds, truncateds, infos = self.vec_env.step(action_to_vec_env) # 執行動作
                    dones = np.logical_or(terminateds, truncateds)

                    '''
                    Moving Average for reward scaling 越近的獎勵權重越大
                    計算每個環境的折扣回報的標準差，再將當前獲得的獎勵除以標準差來進行縮放
                    這樣可以確保獎勵保持穩定，避免獎勵變化過大導致訓練不穩定
                    '''
                    self.discounted_returns = self.hp.GAMMA * self.discounted_returns * (1 - dones) + raw_rewards 
                    self.reward_scaler.update(self.discounted_returns)
                    scaled_rewards = raw_rewards / (self.reward_scaler.std() + 1e-8)

                    if 'global_height' in infos: self.best_height_reached_overall = max(self.best_height_reached_overall, float(np.max(infos['global_height'])))
                    accumulated_rewards += scaled_rewards # frame skip 後的累積獎勵
                    final_dones = np.logical_or(final_dones, dones)
                    self.total_env_steps_collected += self.hp.NUM_ENVS

                    # 紀錄 episode 資訊
                    if "episode" in infos and np.any(infos.get("_episode")):
                        ep_rewards = infos["episode"]["r"][infos["_episode"]]
                        ep_lengths = infos["episode"]["l"][infos["_episode"]]
                        for r, l in zip(ep_rewards, ep_lengths):
                            self.completed_episode_rewards.append(r)
                            self.completed_episode_lengths.append(l)

                    # 截斷的時候檢查最高高度
                    if "_final_info" in infos and np.any(infos.get("_final_info")):
                        for info in infos["final_info"][infos["_final_info"]]:
                            if info and 'global_height' in info and info['global_height'] > self.best_height_reached_overall:
                                self.best_height_reached_overall = info['global_height']

                    current_image_observations = new_image_observations
                    if np.all(dones): break
                
                self.agent.store_transitions_batch(
                    obs_for_buffer, discrete_actions_batch, 
                    log_probs_batch, accumulated_rewards, final_dones,
                    values_ext_batch
                )

            with torch.no_grad(), autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                _, last_values_ext_np, _ = self.agent.select_action(current_image_observations) # Bootstrapping 如果採樣結束時還沒 done 就用最後的 value 來估計

            avg_actor_loss, avg_critic_loss, avg_entropy = self.agent.update(
                last_values_ext_np, final_dones
            )

            updates += 1
            self.global_update += 1
            
            # log 紀錄
            if self.global_update % self.hp.LOG_INTERVAL_UPDATES == 0:
                avg_rew = np.mean(self.completed_episode_rewards) if self.completed_episode_rewards else 0.0
                avg_len = np.mean(self.completed_episode_lengths) if self.completed_episode_lengths else 0.0
                end_time = time.time() - start_time
                height_to_print = float(self.best_height_reached_overall)
                print(f"更新: {self.global_update} | 關卡: {self.current_training_target_level+1}")
                print(f"  平均獎勵({len(self.completed_episode_rewards)}eps): {avg_rew:.2f} | 長度: {avg_len:.1f} | 最高高度: {height_to_print:.1f}")
                
                loss_str = f"Actor: {avg_actor_loss:.4f}, Critic: {avg_critic_loss:.4f}"
                print(f"  Loss -> {loss_str} | Entropy: {avg_entropy:.4f} | Time: {end_time:.1f}s\n")
                
                log_entry = {
                    'global_update': self.global_update, 'avg_reward': float(f"{avg_rew:.4f}"), 
                    'avg_length': float(f"{avg_len:.1f}"), 'best_height_reached': float(f"{height_to_print:.1f}"), 
                    'avg_entropy': float(f"{avg_entropy:.4f}"), 'avg_actor_loss': float(f"{avg_actor_loss:.4f}"), 
                    'avg_critic_loss': float(f"{avg_critic_loss:.4f}")
                }
                self.training_log_data.append(log_entry)
                self.save_training_log()

            if self.global_update > 0 and self.global_update % self.hp.SAVE_INTERVAL_UPDATES == 0: 
                self.save_checkpoint()
            
        self.save_checkpoint()
        print(f"--- 目標關卡 {self.current_training_target_level + 1} 訓練完成。---")
        self.vec_env.close()
        pygame.quit()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 PPO 和自訂 CNN 訓練 Jump King Agent。")
    parser.add_argument("--target_level", type=int, default=1, help="要訓練的目標關卡編號 (從 1 開始)。")
    parser.add_argument("--base_model_level", type=int, default=None, help="可選：要作為訓練起點的模型是來自於哪一關。")
    
    args = parser.parse_args()
    
    hyperparameters = Hyperparameters()
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    if not os.path.exists(CHECKPOINT_PATH): os.makedirs(CHECKPOINT_PATH)
        
    trainer = JumpKingPPO_Sync( 
        hyperparams=hyperparameters, user_specified_target_level=args.target_level, 
        base_model_level=args.base_model_level
    )
    trainer.run_training_loop()
