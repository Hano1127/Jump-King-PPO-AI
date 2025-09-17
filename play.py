import pygame
import numpy as np
import os
from collections import deque
import time

pygame.init()
pygame.display.init()
pygame.font.init()

# 從其他檔案導入設定與類別
from settings import *
from player import Player
from levelSetupFunction import MAP_LINES
from ppo_agent import PPOAgent

# 主要的遊戲播放類別
class JumpKingPlay:
    def __init__(self, initial_model_file=None):
        # 初始化
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Jump King (載入中...)")
        self.font = pygame.font.Font(None, 30)

        load_all_images()
        Player.load_sprites()

        self.levels = MAP_LINES
        self.show_lines = False # 是否顯示碰撞線 (偵錯用)
        self.manual_play_mode = False # 是否為手動遊玩模式
        self.clock = pygame.time.Clock()

        self.player = Player()

        num_channels = 3
        self.observation_shape = (FRAME_STACK_SIZE, IMAGE_INPUT_H, IMAGE_INPUT_W, num_channels)
        self.action_dims = [3, 2]

        # 預先載入資源以提升效能
        self.preloaded_level_backgrounds = {}
        self._preload_all_level_backgrounds()
        self.loaded_agents = {}
        self._preload_all_models()
        
        # 建立一個給 AI 觀測用的小畫布
        self.ai_observation_canvas = pygame.Surface((IMAGE_INPUT_W, IMAGE_INPUT_H))
        
        # 用於優化渲染的 Surface
        self.background_surface = pygame.Surface((WIDTH, HEIGHT))
        self.last_player_rect = None
        self.last_info_rects = []

        self.agent = None
        self.current_model_level = -1
        
        # 用於儲存連續畫面的 Frame Stack
        self.frame_stack = deque(maxlen=FRAME_STACK_SIZE)

        # 根據提供的模型檔案名稱來決定初始關卡
        initial_level_to_load = 0
        if initial_model_file:
            try:
                # 從檔名解析關卡編號
                level_from_filename = int(initial_model_file.split('_')[-1].split('.')[0]) - 1
                initial_level_to_load = level_from_filename
            except (ValueError, IndexError):
                print(f"檔名 '{initial_model_file}' 格式錯誤")

        # 重設玩家位置到初始關卡
        self.player.resetPlayer(initial_level_to_load)
        if 0 <= initial_level_to_load < len(PREDEFINED_SPAWN_POINTS):
            self.player.x, self.player.y = PREDEFINED_SPAWN_POINTS[initial_level_to_load]

        # 初始化 AI 模型、關卡資源和 Frame Stack
        self._check_and_switch_model() 
        self._switch_level_assets(self.player.currentLevelNo)
        self._reset_frame_stack()
        
    # 預先載入所有關卡的背景圖
    def _preload_all_level_backgrounds(self):
        for level_idx, level_data in enumerate(self.levels):
            if level_idx >= MAX_LEVELS: break
            if level_data is None: continue
            try:
                self.preloaded_level_backgrounds[level_idx] = level_data.get_background()
            except Exception as e:
                print(f"載入關卡 {level_idx + 1} 背景時發生錯誤: {e}")

    # 預先載入所有訓練好的 AI 模型
    def _preload_all_models(self):
        for level_idx in range(MAX_LEVELS):
            model_filename = f"ppo_model_level_{level_idx + 1}.pth"
            model_full_path = os.path.join(SAVE_PATH, model_filename)
            if not os.path.exists(model_full_path): continue
            # 建立一個新的 Agent 並載入模型權重
            agent = PPOAgent(self.observation_shape, self.action_dims, n_steps=1, num_envs=1, total_training_updates=1)
            agent.load_model(model_full_path)
            self.loaded_agents[level_idx] = agent

    # 切換關卡時，更換背景圖
    def _switch_level_assets(self, level_idx):
        if level_idx in self.preloaded_level_backgrounds:
            self.background_surface.blit(self.preloaded_level_backgrounds[level_idx], (0, 0))
        else:
            self.background_surface.fill((50, 50, 50))
        self.window.blit(self.background_surface, (0, 0))
        pygame.display.flip()
        # 清除舊的繪製區域，來強制重繪整個畫面
        self.last_player_rect = None
        self.last_info_rects = []

    # 將當前遊戲畫面渲染到 AI 觀測用的小畫布上
    def _render_to_scaled_canvas(self, canvas):
        level_idx_to_draw = self.player.currentLevelNo
        level_to_draw = self.levels[level_idx_to_draw] if 0 <= level_idx_to_draw < len(self.levels) and self.levels[level_idx_to_draw] is not None else None
        
        if level_to_draw:
            bg_image = level_to_draw.get_scaled_background()
            canvas.blit(bg_image, (0, 0))
        else:
            canvas.fill((50, 50, 50))

        img_to_draw, blit_pos = self.player.get_scaled_draw_info()
        if img_to_draw and blit_pos:
            canvas.blit(img_to_draw, blit_pos)

    # 獲取單一幀的觀測畫面
    def _get_single_frame_observation(self):
        self._render_to_scaled_canvas(self.ai_observation_canvas)

        # 在 AI 畫布上繪製蓄力條
        charge_value = self.player.jumpTimer
        if charge_value > 0:
            max_bar_width = int(IMAGE_INPUT_W * CHARGE_BAR_WIDTH_RATIO)
            bar_width = int((charge_value / MAX_JUMP_TIMER) * max_bar_width)
            
            bar_x = (IMAGE_INPUT_W - bar_width) // 2
            bar_y = IMAGE_INPUT_H - CHARGE_BAR_HEIGHT - CHARGE_BAR_Y_OFFSET

            pygame.draw.rect(self.ai_observation_canvas, CHARGE_BAR_COLOR, (bar_x, bar_y, bar_width, CHARGE_BAR_HEIGHT))

        # 將 Pygame Surface 轉換為 Numpy 陣列
        frame_rgb = pygame.surfarray.array3d(self.ai_observation_canvas)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2)) # (W, H, C) -> (H, W, C)
        return frame_rgb.astype(np.uint8)

    # 重設 Frame Stack，用目前的畫面填滿
    def _reset_frame_stack(self):
        self.frame_stack.clear()
        initial_frame = self._get_single_frame_observation()
        for _ in range(FRAME_STACK_SIZE):
            self.frame_stack.append(initial_frame)

    # 取得堆疊後的多幀畫面作為 AI 的輸入
    def _get_stacked_observation(self):
        return np.array(self.frame_stack)

    # 檢查並根據玩家目前所在的關卡切換 AI 模型
    def _check_and_switch_model(self):
        new_level_idx = self.player.currentLevelNo
        if new_level_idx == self.current_model_level:
            return 

        target_agent = None
        target_level = -1

        # 如果有對應當前關卡的模型，就直接使用
        if new_level_idx in self.loaded_agents:
            target_agent = self.loaded_agents[new_level_idx]
            target_level = new_level_idx
        else:
            # 如果沒有，則尋找一個最接近且小於等於當前關卡的模型作為替代
            available_levels = sorted(self.loaded_agents.keys())
            if not available_levels:
                return # 沒有任何模型可用

            # 找出所有比當前關卡低的有效模型
            valid_fallbacks = [lvl for lvl in available_levels if lvl <= new_level_idx]
            if valid_fallbacks:
                fallback_level = max(valid_fallbacks) 
            else:
                fallback_level = available_levels[0] 

            if fallback_level != self.current_model_level:
                target_agent = self.loaded_agents[fallback_level]
                target_level = fallback_level

        if target_agent is not None and self.agent != target_agent:
            self.agent = target_agent
            self.current_model_level = target_level
        
        pygame.display.set_caption(f"Jump King (模型: Lvl {self.current_model_level + 1})")


    # 主遊戲循環
    def run_game_loop(self):
        running = True
        action_decision_counter = 0 # Frame skip 計數器
        last_ai_action = (0, 0) # 上一次 AI 的決策

        while running:
            dirty_rects = [] # 記錄畫面中需要更新的區域
            player_level_before_update = self.player.currentLevelNo

            # 事件處理
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB: self.manual_play_mode = not self.manual_play_mode
                    elif event.key == pygame.K_l: self.show_lines = not self.show_lines
                    elif event.key == pygame.K_ESCAPE: running = False
                    elif event.key == pygame.K_x: print(f"x:{self.player.x:.1f}, y:{self.player.y:.1f}") # 顯示玩家座標
                    elif event.key in [pygame.K_n, pygame.K_p]: # 切換關卡 (N:下一關, P:上一關)
                        if event.key == pygame.K_n:
                            next_level_idx = (self.player.currentLevelNo + 1) % MAX_LEVELS
                        else:
                            next_level_idx = (self.player.currentLevelNo - 1 + MAX_LEVELS) % MAX_LEVELS
                        self.player.resetPlayer(next_level_idx)
                        if 0 <= next_level_idx < len(PREDEFINED_SPAWN_POINTS): self.player.x, self.player.y = PREDEFINED_SPAWN_POINTS[next_level_idx]
                        self._check_and_switch_model()
                        self._reset_frame_stack()

            # 模式
            if self.manual_play_mode:
                # 手動模式
                pressed_keys = pygame.key.get_pressed()
                self.player.leftHeld = pressed_keys[pygame.K_LEFT]
                self.player.rightHeld = pressed_keys[pygame.K_RIGHT]
                self.player.jumpHeld = pressed_keys[pygame.K_SPACE]
            else:
                # AI 模式
                if action_decision_counter == 0 and self.agent:
                    stacked_obs = self._get_stacked_observation()
                    #start_time = time.perf_counter()

                    discrete_actions, _, _= self.agent.select_action(
                        np.array([stacked_obs]),
                        deterministic=True # 測試階段，選擇機率最高的動作
                    )
                    #end_time = time.perf_counter()
                    #decision_time_ms = (end_time - start_time) * 1000
                    # print(f"decision: {decision_time_ms:.2f} ms")
                    action_parts = discrete_actions[0]
                    last_ai_action = (action_parts[0], action_parts[1])

                # 根據 AI 的決策來控制玩家
                jump_action_idx, direction_action_idx = last_ai_action
                
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

                action_decision_counter = (action_decision_counter + 1) % ACTION_DECISION_INTERVAL

            # 更新玩家狀態
            self.player.Update(single_mode=True)
            
            # 獲取新的一幀並加入 Frame Stack
            new_frame = self._get_single_frame_observation()
            self.frame_stack.append(new_frame)

            if self.player.players_dead:
                self.player.resetPlayer(self.player.currentLevelNo)
                self._reset_frame_stack()

            if self.player.currentLevelNo != player_level_before_update:
                self._switch_level_assets(self.player.currentLevelNo)
                self._reset_frame_stack()
                continue # 跳過這一幀的後續渲染，因為背景已在切換函式中重繪
            
            # 如果玩家跳到新的一關，檢查是否需要更換模型
            if self.player.stable_on_new_level_up:
                self._check_and_switch_model()
            
            # 清除上一幀資訊佔據的區域
            for r in self.last_info_rects: self.window.blit(self.background_surface, r.topleft, r)
            dirty_rects.extend(self.last_info_rects)
            self.last_info_rects.clear()
            if self.last_player_rect:
                self.window.blit(self.background_surface, self.last_player_rect.topleft, self.last_player_rect)
                dirty_rects.append(self.last_player_rect)
            
            # 繪製新一幀的玩家
            player_sprite, player_pos, player_rect = self.player.get_draw_info()
            if player_sprite:
                self.window.blit(player_sprite, player_pos)
                dirty_rects.append(player_rect)
                self.last_player_rect = player_rect
            
            # 只更新畫面中有變動的區域
            pygame.display.update(dirty_rects)

            # 控制遊戲幀率
            self.clock.tick(FPS)
            
        pygame.quit()

if __name__ == "__main__":
    game_player = JumpKingPlay()
    game_player.run_game_loop()
