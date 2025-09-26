import pygame
import numpy as np
import os
from collections import deque
import time

pygame.init()
pygame.display.init()
pygame.font.init()

# Import settings and classes from other files
from settings import *
from player import Player
from levelSetupFunction import MAP_LINES
from ppo_agent import PPOAgent

# Main game playback class
class JumpKingPlay:
    def __init__(self, initial_model_file=None):
        # Initialization
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Jump King (Loading...)")
        self.font = pygame.font.Font(None, 30)

        load_all_images()
        Player.load_sprites()

        self.levels = MAP_LINES
        self.show_lines = False # Whether to show collision lines (for debugging)
        self.manual_play_mode = False # Whether to enable manual play mode
        self.clock = pygame.time.Clock()

        self.player = Player()

        num_channels = 3
        self.observation_shape = (FRAME_STACK_SIZE, IMAGE_INPUT_H, IMAGE_INPUT_W, num_channels)
        self.action_dims = [3, 2]

        # Preload resources to improve performance
        self.preloaded_level_backgrounds = {}
        self.preload_all_level_backgrounds()
        self.loaded_agents = {}
        self.preload_all_models()
        
        # Create a small canvas for AI observation
        self.ai_observation_canvas = pygame.Surface((IMAGE_INPUT_W, IMAGE_INPUT_H))
        
        # Surface for rendering optimization
        self.background_surface = pygame.Surface((WIDTH, HEIGHT))
        self.last_player_rect = None
        self.last_info_rects = []

        self.agent = None
        self.current_model_level = -1
        
        # Frame Stack for storing consecutive frames
        self.frame_stack = deque(maxlen=FRAME_STACK_SIZE)

        # Determine the initial level based on the provided model file name
        initial_level_to_load = 0
        if initial_model_file:
            level_from_filename = int(initial_model_file.split('_')[-1].split('.')[0]) - 1
            initial_level_to_load = level_from_filename

        # Reset player position to the initial level
        self.player.resetPlayer(initial_level_to_load)
        if 0 <= initial_level_to_load < len(PREDEFINED_SPAWN_POINTS):
            self.player.x, self.player.y = PREDEFINED_SPAWN_POINTS[initial_level_to_load]

        # Initialize the AI model, level assets, and Frame Stack
        self.check_and_switch_model() 
        self._switch_level_assets(self.player.currentLevelNo)
        self.reset_frame_stack()
        
    # Preload backgrounds for all levels
    def preload_all_level_backgrounds(self):
        for level_idx, level_data in enumerate(self.levels):
            if level_idx >= MAX_LEVELS: break
            if level_data is None: continue
            try:
                self.preloaded_level_backgrounds[level_idx] = level_data.get_background()
            except Exception as e:
                print(f"Error loading background for level {level_idx + 1}: {e}")

    # Preload all trained AI models
    def preload_all_models(self):
        for level_idx in range(MAX_LEVELS):
            model_filename = f"ppo_model_level_{level_idx + 1}.pth"
            model_full_path = os.path.join(SAVE_PATH, model_filename)
            if not os.path.exists(model_full_path): continue
            # Create a new Agent and load the model weights
            agent = PPOAgent(self.observation_shape, self.action_dims, n_steps=1, num_envs=1, total_training_updates=1)
            agent.load_model(model_full_path)
            self.loaded_agents[level_idx] = agent

    # Switch background image when changing levels
    def _switch_level_assets(self, level_idx):
        if level_idx in self.preloaded_level_backgrounds:
            self.background_surface.blit(self.preloaded_level_backgrounds[level_idx], (0, 0))
        else:
            self.background_surface.fill((50, 50, 50))
        self.window.blit(self.background_surface, (0, 0))
        pygame.display.flip()
        # Clear old drawing areas to force a full redraw
        self.last_player_rect = None
        self.last_info_rects = []

    # Render the current game screen to the small canvas for AI observation
    def render_to_scaled_canvas(self, canvas):
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

    # Get a single frame observation
    def get_single_frame_observation(self):
        self.render_to_scaled_canvas(self.ai_observation_canvas)

        # Draw the charge bar on the AI canvas
        charge_value = self.player.jumpTimer
        if charge_value > 0:
            max_bar_width = int(IMAGE_INPUT_W * CHARGE_BAR_WIDTH_RATIO)
            bar_width = int((charge_value / MAX_JUMP_TIMER) * max_bar_width)
            
            bar_x = (IMAGE_INPUT_W - bar_width) // 2
            bar_y = IMAGE_INPUT_H - CHARGE_BAR_HEIGHT - CHARGE_BAR_Y_OFFSET

            pygame.draw.rect(self.ai_observation_canvas, CHARGE_BAR_COLOR, (bar_x, bar_y, bar_width, CHARGE_BAR_HEIGHT))

        # Convert Pygame Surface to a Numpy array
        frame_rgb = pygame.surfarray.array3d(self.ai_observation_canvas)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2)) # (W, H, C) -> (H, W, C)
        return frame_rgb.astype(np.uint8)

    # Reset the Frame Stack, filling it with the current frame
    def reset_frame_stack(self):
        self.frame_stack.clear()
        initial_frame = self.get_single_frame_observation()
        for _ in range(FRAME_STACK_SIZE):
            self.frame_stack.append(initial_frame)

    # Get the stacked frames as input for the AI
    def _get_stacked_observation(self):
        return np.array(self.frame_stack)

    # Check and switch the AI model based on the player's current level
    def check_and_switch_model(self):
        new_level_idx = self.player.currentLevelNo
        if new_level_idx == self.current_model_level:
            return 

        target_agent = None
        target_level = -1

        # If a model for the current level exists, use it
        if new_level_idx in self.loaded_agents:
            target_agent = self.loaded_agents[new_level_idx]
            target_level = new_level_idx
        else:
            # Otherwise, find the closest model for a level less than or equal to the current one as a fallback
            available_levels = sorted(self.loaded_agents.keys())
            if not available_levels:
                return # No models available

            # Find all valid fallback models for levels lower than or equal to the current one
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
        
        pygame.display.set_caption(f"Jump King (Model: Lvl {self.current_model_level + 1})")


    # Main game loop
    def run_game_loop(self):
        running = True
        action_decision_counter = 0 # Frame skip counter
        action = (0, 0) # Last AI action decision

        while running:
            dirty_rects = [] # Record the areas of the screen that need to be updated
            player_level_before_update = self.player.currentLevelNo

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB: self.manual_play_mode = not self.manual_play_mode
                    elif event.key == pygame.K_l: self.show_lines = not self.show_lines
                    elif event.key == pygame.K_ESCAPE: running = False
                    elif event.key == pygame.K_x: print(f"x:{self.player.x:.1f}, y:{self.player.y:.1f}") # Print player coordinates
                    elif event.key in [pygame.K_n, pygame.K_p]: # Switch level (N: next, P: previous)
                        if event.key == pygame.K_n:
                            next_level_idx = (self.player.currentLevelNo + 1) % MAX_LEVELS
                        else:
                            next_level_idx = (self.player.currentLevelNo - 1 + MAX_LEVELS) % MAX_LEVELS
                        self.player.resetPlayer(next_level_idx)
                        if 0 <= next_level_idx < len(PREDEFINED_SPAWN_POINTS): self.player.x, self.player.y = PREDEFINED_SPAWN_POINTS[next_level_idx]
                        self.check_and_switch_model()
                        self.reset_frame_stack()

            # Game Mode
            if self.manual_play_mode:
                # Manual mode
                pressed_keys = pygame.key.get_pressed()
                self.player.leftHeld = pressed_keys[pygame.K_LEFT]
                self.player.rightHeld = pressed_keys[pygame.K_RIGHT]
                self.player.jumpHeld = pressed_keys[pygame.K_SPACE]
            else:
                # AI mode
                if action_decision_counter == 0 and self.agent:
                    stacked_obs = self._get_stacked_observation()
                    #start_time = time.perf_counter()

                    discrete_actions, _, _= self.agent.select_action(
                        np.array([stacked_obs]),
                        deterministic=True # In testing phase, select the most likely action
                    )
                    #end_time = time.perf_counter()
                    #decision_time_ms = (end_time - start_time) * 1000
                    # print(f"decision: {decision_time_ms:.2f} ms")
                    action_parts = discrete_actions[0]
                    action = (action_parts[0], action_parts[1])

                # Control the player based on the AI's decision
                jump_idx, direction_idx = action
                
                if jump_idx == 1: 
                    if self.player.isOnGround:
                        self.player.jumpHeld = True
                elif jump_idx == 2: 
                    if self.player.jumpHeld:
                        self.player.jumpHeld = False
                        self.player.jump_released = False
                
                if direction_idx == 0: 
                    self.player.leftHeld = True
                    self.player.rightHeld = False
                elif direction_idx == 1: 
                    self.player.leftHeld = False
                    self.player.rightHeld = True

                action_decision_counter = (action_decision_counter + 1) % ACTION_DECISION_INTERVAL

            # Update player state
            self.player.Update(single_mode=True)
            
            # Get a new frame and add it to the Frame Stack
            new_frame = self.get_single_frame_observation()
            self.frame_stack.append(new_frame)

            if self.player.players_dead:
                self.player.resetPlayer(self.player.currentLevelNo)
                self.reset_frame_stack()

            if self.player.currentLevelNo != player_level_before_update:
                self._switch_level_assets(self.player.currentLevelNo)
                self.reset_frame_stack()
                continue # Skip rendering for this frame, as the background was redrawn in the switch function
            
            # If the player reaches a new level, check if the model needs to be switched
            if self.player.stable_on_new_level_up:
                self.check_and_switch_model()
            
            # Clear the areas occupied by the previous frame's elements
            for r in self.last_info_rects: self.window.blit(self.background_surface, r.topleft, r)
            dirty_rects.extend(self.last_info_rects)
            self.last_info_rects.clear()
            if self.last_player_rect:
                self.window.blit(self.background_surface, self.last_player_rect.topleft, self.last_player_rect)
                dirty_rects.append(self.last_player_rect)
            
            # Draw the new player sprite
            player_sprite, player_pos, player_rect = self.player.get_draw_info()
            if player_sprite:
                self.window.blit(player_sprite, player_pos)
                dirty_rects.append(player_rect)
                self.last_player_rect = player_rect
            
            # Update only the changed areas of the screen
            pygame.display.update(dirty_rects)

            # Control the game's frame rate
            self.clock.tick(FPS)
            
        pygame.quit()

if __name__ == "__main__":
    game_player = JumpKingPlay()
    game_player.run_game_loop()
