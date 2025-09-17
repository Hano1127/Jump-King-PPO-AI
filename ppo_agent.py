import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from settings import (IMAGE_INPUT_H, IMAGE_INPUT_W, USE_GRAYSCALE, FRAME_STACK_SIZE, CONTRAST_FACTOR) 

from torch.amp import GradScaler
from utils import init_weights_orthogonal
from torchvision.transforms.functional import rgb_to_grayscale, adjust_contrast


# 殘差區塊
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 兩層卷積層穿插批次正規
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 捷徑
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主要的輸出
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.bn1(self.conv1(out))
        
        # 將輸出與捷徑輸出相加
        out += self.shortcut(x)

        out = F.relu(out)
        return out


class ActorCritic(nn.Module):
    def __init__(self, action_dims):

        # 繼承 nn.Module
        super(ActorCritic, self).__init__()

        # 動作維度
        self.action_dims = action_dims

        # 輸入維度
        in_channels = FRAME_STACK_SIZE * (1 if USE_GRAYSCALE else 3)
        
        # CNN Resnet
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),

            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),

            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        # 計算輸出維度
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, IMAGE_INPUT_H, IMAGE_INPUT_W)
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        # MLP 多層感知機
        mlp_input_dim = cnn_output_dim 
        mlp_hidden_size = 512
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )

        # Actor-Critic 頭部
        self.actor_discrete_heads = nn.ModuleList([nn.Linear(mlp_hidden_size, dim) for dim in action_dims])
        self.critic_ext = nn.Linear(mlp_hidden_size, 1)

        # 正交初始化 
        # 高斯分布初始化權重 -> 奇異值分解 -> 用正交矩陣做權重矩陣
        self.apply(lambda m: init_weights_orthogonal(m, gain=np.sqrt(2)))
        for head in self.actor_discrete_heads:
            head.apply(lambda m: init_weights_orthogonal(m, gain=0.01))
        self.critic_ext.apply(lambda m: init_weights_orthogonal(m, gain=1.0))
    
    def get_action_and_value(self, preprocessed_image_state, discrete_actions_input=None, deterministic=False):
        
        # 將處理過的畫面送入CNN
        cnn_features = self.cnn(preprocessed_image_state)
        
        # 將CNN的特徵送入MLP
        mlp_out = self.mlp(cnn_features)
        
        # 將 MLP 的輸出傳入 Actor 的各個頭部
        discrete_logits = [head(mlp_out) for head in self.actor_discrete_heads]

        # 將 discrete_logits 轉換成一個個可採樣的機率分佈
        discrete_distributions = [Categorical(logits=logits) for logits in discrete_logits]
        
        # 將 MLP 的輸出傳入 Critic 網路
        value = self.critic_ext(mlp_out)

        if discrete_actions_input is None: # 如果正在進行動作採樣
            if not deterministic: # 採樣階段照機率採樣動作
                actions_tensors = [dist.sample() for dist in discrete_distributions] 
            else: # 測試階段選擇機率最高的動作
                actions_tensors = [torch.argmax(dist.logits, dim=-1) for dist in discrete_distributions]
        else: 
            actions_tensors = discrete_actions_input

        # 計算對數機率
        log_probs = sum(dist.log_prob(act) for dist, act in zip(discrete_distributions, actions_tensors))

        # 計算熵
        entropy = sum(dist.entropy() for dist in discrete_distributions).mean()

        return actions_tensors, log_probs, entropy, value

class PPOAgent:
    def __init__(self, image_observation_shape, action_dims, n_steps, num_envs, total_training_updates, lr=3e-4, gamma=0.99, n_epochs=4, eps_clip=0.2, entropy_coeff=0.01, gae_lambda=0.95, adam_eps=1e-5, mini_batch_size=256):
        
        # 初始化
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.n_epochs = n_epochs
        self.entropy_coeff = entropy_coeff
        self.gae_lambda = gae_lambda
        self.image_observation_shape = image_observation_shape
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.num_discrete_action_dims = len(action_dims)
        self.mini_batch_size = mini_batch_size
        
        self.current_update_num = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            print(f" 使用設備: {self.device}")

        self.policy = ActorCritic(action_dims).to(self.device)
        self.policy_old = ActorCritic(action_dims).to(self.device)

        if int(torch.__version__.split('.')[0]) >= 2:
            try: 
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
                self.policy_old = torch.compile(self.policy_old, mode="reduce-overhead")
            except Exception as e:
                print(f"嘗試 torch.compile 失敗: {e}")

        # Adam 優化器: 保留梯度且做加權以更新的更好更快
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=adam_eps)

        # 學習率線性衰減 
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_training_updates)
        
        # 複製策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 均方誤差
        self.MseLoss = nn.MSELoss()

        # 自動化縮放梯度 
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
                
        self._setup_buffer()

    def _setup_buffer(self):
        self.image_states = np.zeros((self.n_steps, self.num_envs, *self.image_observation_shape), dtype=np.uint8)
        self.discrete_actions = np.zeros((self.n_steps, self.num_envs, self.num_discrete_action_dims), dtype=np.int64)
        self.log_probs = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.num_envs), dtype=bool)
        self.values_ext = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.step_ptr = 0

    def clear_buffer(self):
        self.step_ptr = 0
    
    # 預處理圖像
    def _preprocess_obs_batch(self, obs_batch_numpy): 
        obs_tensor = torch.from_numpy(obs_batch_numpy).to(self.device, non_blocking=True) # 將數據轉成 Pytorch Tensor 然後移動到 GPU 上
        obs_tensor = obs_tensor.permute(0, 1, 4, 2, 3) # 修改資料順序
        obs_tensor = obs_tensor.float().div_(255.0) # 轉成 float 並且正規化

        N, S, C, H, W = obs_tensor.shape

        if USE_GRAYSCALE and C == 3: # 把畫面轉成灰階
            obs_tensor = rgb_to_grayscale(obs_tensor.view(N * S, C, H, W), num_output_channels=1)
            C = 1

        if CONTRAST_FACTOR != 1.0: # 調整對比度
            obs_tensor = adjust_contrast(obs_tensor, contrast_factor=CONTRAST_FACTOR)

        final_channels = S * C # 將堆疊幀疊起來

        return obs_tensor.view(N, final_channels, H, W)

    # 儲存數據已供更新
    def store_transitions_batch(self, image_state, discrete_actions, log_prob, reward, done, value):
        if self.step_ptr >= self.n_steps: return # 檢查緩衝區是否已滿
        self.image_states[self.step_ptr] = image_state # state
        self.discrete_actions[self.step_ptr] = np.array(discrete_actions) # action
        self.log_probs[self.step_ptr] = log_prob # 動作機率
        self.rewards[self.step_ptr] = reward # reward
        self.dones[self.step_ptr] = done # 是否截斷
        self.values_ext[self.step_ptr] = value.flatten() # 價值估計
        self.step_ptr += 1

    # 決策
    def select_action(self, image_states_batch_numpy, deterministic=False):

        self.policy_old.eval() # 評估模式

        with torch.no_grad(): # 關閉梯度計算
            image_states_tensor = self._preprocess_obs_batch(image_states_batch_numpy) 
            
            actions_tensors, log_probs, _, values_ext = self.policy_old.get_action_and_value(
                image_states_tensor, deterministic=deterministic
            )
            
            actions_np = torch.stack(actions_tensors, dim=1).cpu().numpy() # 把動作合併成一個 tensor 並轉成 numpy 
            values_ext_np = values_ext.cpu().numpy().flatten()
            log_probs_np = log_probs.cpu().numpy()
            
        return actions_np, values_ext_np, log_probs_np

    def update(self, next_values_ext_bootstrap, next_dones_bootstrap):
        if self.step_ptr == 0: return (0.0,) * 3 
        
        self.current_update_num += 1
        
        extrinsic_rewards = self.rewards
        advantages_ext = np.zeros_like(extrinsic_rewards) # 初始化一個跟獎勵形狀一樣的優勢函數陣列
        last_gae_lam_ext = 0 # 上一步的 GAE 值
        full_values_ext = np.vstack([self.values_ext, next_values_ext_bootstrap.reshape(1, -1)])
        full_dones = np.vstack([self.dones, next_dones_bootstrap.reshape(1, -1)])

        for step in reversed(range(self.n_steps)): # 從最後一步開始往前計算 GAE
            next_non_terminal = 1.0 - full_dones[step + 1] # 判斷下一步有沒有終止
            delta = extrinsic_rewards[step] + self.gamma * full_values_ext[step + 1] * next_non_terminal - full_values_ext[step] # TD-error
            advantages_ext[step] = last_gae_lam_ext = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam_ext # GAE 計算

        returns_ext_np = advantages_ext + self.values_ext # A + V = Q

        batch_size = self.n_steps * self.num_envs
        
        # 將 numpy 轉換為 tensor 並調整形狀
        b_image_states_numpy = self.image_states.swapaxes(0, 1).reshape(batch_size, *self.image_observation_shape)
        b_actions = torch.from_numpy(self.discrete_actions.swapaxes(0, 1).reshape(batch_size, self.num_discrete_action_dims)).long().to(self.device)
        b_old_log_probs = torch.from_numpy(self.log_probs.swapaxes(0, 1).flatten()).float().to(self.device)
        
        b_advantages_ext = torch.from_numpy(advantages_ext.swapaxes(0, 1).flatten()).float().to(self.device)
        b_returns_ext = torch.from_numpy(returns_ext_np.swapaxes(0, 1).flatten()).float().to(self.device)

        # 優勢正規化
        b_advantages = (b_advantages_ext - b_advantages_ext.mean()) / (b_advantages_ext.std() + 1e-8)

        self.policy.train() # 訓練模式
        actor_losses, critic_losses, entropy_losses = [], [], []
        indices = np.arange(batch_size) # 樣本的索引

        for _ in range(self.n_epochs): 
            np.random.shuffle(indices) # 打亂索引順序

            # 將數據分成多個 mini_batch 做計算
            for start in range(0, batch_size, self.mini_batch_size): 
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                
                mb_image_states_tensor = self._preprocess_obs_batch(b_image_states_numpy[mb_indices])
                
                # 新網路的計算
                _, new_log_probs, entropy, new_values_ext = self.policy.get_action_and_value( 
                    mb_image_states_tensor, 
                    [b_actions[mb_indices, i] for i in range(self.num_discrete_action_dims)]
                )
                
                # 處理多餘的維度
                new_values_ext = new_values_ext.squeeze()
                
                mb_advantages = b_advantages[mb_indices]
                
                ratios = torch.exp(new_log_probs - b_old_log_probs[mb_indices]) # 計算新舊機率比
                surr1 = ratios * mb_advantages # surrogate objective
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages # clip
                actor_loss = -torch.min(surr1, surr2).mean() 
                
                critic_loss_ext = self.MseLoss(new_values_ext, b_returns_ext[mb_indices]) # 均方誤差損失

                entropy_loss = entropy # 熵損失

                loss = actor_loss + 0.5 * critic_loss_ext - self.entropy_coeff * entropy_loss # 總損失
                
                # 紀錄損失
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss_ext.item())
                entropy_losses.append(entropy_loss.item())
                
                # 反向傳播
                self.optimizer.zero_grad(set_to_none=True) 
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

                # 更新參數、縮放器
                self.scaler.step(self.optimizer)
                self.scaler.update()

        self.scheduler.step() # 學習率衰減
        self.policy_old.load_state_dict(self.policy.state_dict()) # 更新舊策略
        
        return (np.mean(actor_losses), 
                np.mean(critic_losses), 
                np.mean(entropy_losses))

    def save_model(self, filepath):
        model_to_save = self.policy._orig_mod if hasattr(self.policy, '_orig_mod') else self.policy # for torch.compile
        torch.save(model_to_save.state_dict(), filepath)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath, set_to_training_mode=False):
        try:
            state_dict = torch.load(filepath, map_location=self.device)
            
            policy_to_load = self.policy._orig_mod if hasattr(self.policy, '_orig_mod') else self.policy
            policy_old_to_load = self.policy_old._orig_mod if hasattr(self.policy_old, '_orig_mod') else self.policy_old
            
            policy_to_load.load_state_dict(state_dict, strict=True)
            policy_old_to_load.load_state_dict(state_dict, strict=True)
            
            if set_to_training_mode:
                self.policy.train()
                self.policy_old.train()
            else:
                self.policy.eval()
                self.policy_old.eval()
            print(f"成功從 '{filepath}' 載入模型權重 (strict=True)。")

        except Exception as e:
            print(f"載入模型 {filepath} 時發生錯誤: {e}。將使用初始權重。")