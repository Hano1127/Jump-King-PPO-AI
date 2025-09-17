import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from settings import *
import math
from player import Player

# JumpKing 環境

class JumpKingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, initial_target_level=0, render_mode=None, max_episode_steps=3200, shared_levels=None):
        
        # 繼承 gym.Env
        super(JumpKingEnv, self).__init__()

        # 初始化
        self.initial_target_level = initial_target_level
        self._max_episode_steps = max_episode_steps
        self.current_episode_step_count = 0
        self.episode_total_reward = 0.0


        if shared_levels:
            self.levels = shared_levels
        else:
            from levelSetupFunction import MAP_LINES
            self.levels = MAP_LINES

        if not pygame.get_init():
            pygame.init()

        self.player = Player(initial_target_level=self.initial_target_level)
        
        self.visited_platforms = set()

        # 動作空間 
        # 0: 不動 
        # 1: 按住空白鍵 (蓄力)
        # 2: 放開空白鍵

        # 0: 向左
        # 1: 向右
        self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(2)))
        
        # 狀態空間
        channels = 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(IMAGE_INPUT_H, IMAGE_INPUT_W, channels), dtype=np.uint8
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.observation_canvas = pygame.Surface((IMAGE_INPUT_W, IMAGE_INPUT_H), pygame.SRCALPHA)

        if self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.Font(None, 30)
            except pygame.error:
                self.font = pygame.font.SysFont("arial", 28)

    # 設置到關卡
    def set_player_target_level(self, target_level_idx):
        self.initial_target_level = target_level_idx
        if hasattr(self.player, 'set_current_training_target_level'):
            self.player.set_current_training_target_level(target_level_idx)

    # 獲取輸入
    def _get_obs(self):
        # 將遊戲畫面渲染到 AI 的觀測畫布上
        self._render_to_scaled_canvas(self.observation_canvas)

        # 將蓄力值視覺化
        charge_value = self.player.jumpTimer
        if charge_value > 0:
            max_bar_width = int(IMAGE_INPUT_W * CHARGE_BAR_WIDTH_RATIO)
            bar_width = int((charge_value / MAX_JUMP_TIMER) * max_bar_width)
            
            bar_x = (IMAGE_INPUT_W - bar_width) // 2
            bar_y = IMAGE_INPUT_H - CHARGE_BAR_HEIGHT - CHARGE_BAR_Y_OFFSET

            pygame.draw.rect(self.observation_canvas, CHARGE_BAR_COLOR, (bar_x, bar_y, bar_width, CHARGE_BAR_HEIGHT))

        # 把畫面轉換為 Numpy 陣列
        frame_rgb = pygame.surfarray.array3d(self.observation_canvas)

        # 把 (W, H, C) 轉成 (H, W, C)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        
        return frame_rgb.astype(np.uint8)

    # 取得目標座標
    def _get_target_coords(self):
 
        # 目標設為下一關的出生點
        next_level_idx = self.player.current_training_target_level + 1
        if 0 <= next_level_idx < len(PREDEFINED_SPAWN_POINTS):
            target_x, target_y_local_next_level = PREDEFINED_SPAWN_POINTS[next_level_idx]
            target_y_global = (next_level_idx * HEIGHT) + (HEIGHT - target_y_local_next_level)
            return target_x, target_y_global
        
        # 如果沒有下一關，以玩家目前位置為目標
        return self.player.x, self.player.getGlobalHeight()

    # 獎勵函數
    def _calculate_reward(self, prev_info, current_info, done, landed_on_new_platform):
        reward = -0.02 # 時間懲罰

        current_height = current_info['global_height']
        prev_height = prev_info['global_height']
        height_diff = current_height - prev_height

        reward += height_diff * 0.02 # 高度變化獎勵/懲罰

        if landed_on_new_platform and len(self.visited_platforms) > 1: 
            reward += 2.5 # 探索到新的平台

        if done:
            if current_info.get('stable_on_new_level_up', False): 
                reward += 20.0 # 跳到下一關
            elif current_info.get('felt_to_previous_level', False):
                reward -= 5.0 # 掉回上一關
            elif current_info.get('players_dead', False):
                reward -= 25.0
                print("BUG") 
        return np.clip(reward, -25.0, 25.0) # 限制獎勵範圍

    # 執行決策
    def step(self, action):

        jump_action_idx, direction_action_idx = action
        
        if jump_action_idx == 1:
            if self.player.isOnGround:
                self.player.jumpHeld = True

        elif jump_action_idx == 2:
            if self.player.jumpHeld:
                self.player.jumpHeld = False
                self.player.space_bar_processed_for_jump = False
        
        if direction_action_idx == 0: 
            self.player.leftHeld = True
            self.player.rightHeld = False
        elif direction_action_idx == 1: 
            self.player.leftHeld = False
            self.player.rightHeld = True
        
        prev_global_height = self.player.getGlobalHeight()
        target_x_before, target_y_global_before = self._get_target_coords()
        prev_dist = math.hypot(self.player.x - target_x_before, prev_global_height - target_y_global_before)
        # 上一步的資訊
        prev_info = {'global_height': prev_global_height, # 高度
                     'dist_to_target': prev_dist, # 與下一關距離
                     'is_on_ground': self.player.isOnGround, # 是否在地上
                     'hasBumped': self.player.hasBumped # 撞牆
                     }

        # 更新遊戲環境
        self.player.Update(single_mode=False)

        # 檢查有沒有探索到新平台
        landed_on_new_platform = False
        if self.player.last_landed_line_id is not None:
            if self.player.last_landed_line_id not in self.visited_platforms:
                landed_on_new_platform = True
                self.visited_platforms.add(self.player.last_landed_line_id)

        current_global_height = self.player.getGlobalHeight()
        target_x_after, target_y_global_after = self._get_target_coords()
        current_dist = math.hypot(self.player.x - target_x_after, current_global_height - target_y_global_after)
        # 這一步的資訊
        current_info = {
            'global_height': current_global_height, 'dist_to_target': current_dist,
            'is_on_ground': self.player.isOnGround, 'stable_on_new_level_up': self.player.stable_on_new_level_up, # 是否到達下一關
            'players_dead': self.player.players_dead, 'felt_to_previous_level': self.player.felt_to_previous_level, # 是否掉到上一關
            'player_x': self.player.x, 'player_y': self.player.y, 'hasBumped': self.player.hasBumped
        }
        
        # 失敗條件
        failed = self.player.currentLevelNo < self.player.current_training_target_level

        # 結束條件
        terminated = self.player.players_dead or self.player.stable_on_new_level_up or failed
        
        # 當前環境的步數
        self.current_episode_step_count += 1

        # 截斷條件
        truncated = self.current_episode_step_count >= self._max_episode_steps
        done = terminated or truncated

        # 計算環境一步的獎勵
        reward = self._calculate_reward(prev_info, current_info, done, landed_on_new_platform)

        # 累計獎勵
        self.episode_total_reward += reward

        # 獲得輸入
        observation = self._get_obs()
        
        # 作為 log 的輸入
        info = {
            'global_height': self.player.getGlobalHeight(),
        }

        # 更新畫面
        if self.render_mode == "human": 
            self.render()
            
        return observation, reward, terminated, truncated, info

    # 重製
    def reset(self, seed=None, options=None):

        # 初始化
        super().reset(seed=seed)
        self.current_episode_step_count = 0
        self.episode_total_reward = 0.0
        self.visited_platforms.clear()
        
        start_level_override = self.initial_target_level
        if options and "start_level" in options: 
            start_level_override = options["start_level"]
        
        # 確保關卡範圍
        start_level_override = max(0, min(start_level_override, len(PREDEFINED_SPAWN_POINTS) - 1))
        
        # 重置到訓練關卡
        if hasattr(self.player, 'set_current_training_target_level'): 
            self.player.set_current_training_target_level(start_level_override) 
        
        # 重製玩家
        self.player.resetPlayer(initial_target_level=start_level_override)
        
        # 設置出生點並加入一點隨機性
        if 0 <= start_level_override < len(PREDEFINED_SPAWN_POINTS):
            spawn_x, spawn_y = PREDEFINED_SPAWN_POINTS[start_level_override]
            random_offset = (np.random.rand() - 0.5) * 10
            self.player.x = spawn_x + random_offset
            self.player.y = spawn_y
        
        # 儲存一個偵錯用的畫面 
        self.player.jumpTimer = 18 # 設定為一半的蓄力值 (暫時)
        observation = self._get_obs()
        self.player.jumpTimer = 0 # 立刻重設，避免影響後續訓練

        info = {'global_height': self.player.getGlobalHeight()}
        if self.render_mode == "human": self.render()

        return observation, info

    # 把圖片繪製到小畫布上
    def _render_to_scaled_canvas(self, canvas):
        # 設定關卡
        level_idx_to_draw = self.player.currentLevelNo
        level_to_draw = self.levels[level_idx_to_draw] if 0 <= level_idx_to_draw < len(self.levels) else None
        if level_to_draw:
            bg_image = level_to_draw.get_scaled_background()
            canvas.blit(bg_image, (0, 0))
        else:
            canvas.fill((50, 50, 50))
        img_to_draw, blit_pos = self.player.get_scaled_draw_info()
        if img_to_draw and blit_pos:
            canvas.blit(img_to_draw, blit_pos)

    # 繪製遊戲畫面 (偵錯用)
    def render(self):
        if self.render_mode == "human":
            if self.window is None: return
            
            # 設定關卡
            level_idx_to_draw = self.player.currentLevelNo
            level_to_draw = self.levels[level_idx_to_draw] if 0 <= level_idx_to_draw < len(self.levels) else None
            if level_to_draw:
                self.window.blit(level_to_draw.get_background(), (0, 0)) # 繪製背景
            else:
                self.window.fill((50, 50, 50))
            self.player.Draw(self.window, single_mode=True) # 繪製玩家

            if self.font:
                info_text = f"Lvl: {self.player.currentLevelNo + 1} H: {self.player.getGlobalHeight():.0f} R: {self.episode_total_reward:.2f}" # 偵錯用的顯示關卡、高度與獎勵
                text_surface = self.font.render(info_text, True, (255, 255, 255))
                self.window.blit(text_surface, (10, 10))

            pygame.event.pump() # 檢查是否有新事件發生
            pygame.display.flip() # 顯示到螢幕上

            self.clock.tick(self.metadata['render_fps'])

    # 關閉
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()
