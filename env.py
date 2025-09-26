import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from settings import *
import math
from player import Player

# JumpKing Environment

class JumpKingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, initial_target_level=0, render_mode=None, max_episode_steps=3200, shared_levels=None):
        
        # Inherit from gym.Env
        super(JumpKingEnv, self).__init__()

        # Initialization
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

        # Action Space 
        # Jump action:
        # 0: No action
        # 1: Hold spacebar (charge jump)
        # 2: Release spacebar
        # Direction action:
        # 0: Left
        # 1: Right
        self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(2)))
        
        # Observation Space
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

    # Set the target level
    def set_player_target_level(self, target_level_idx):
        self.initial_target_level = target_level_idx
        if hasattr(self.player, 'set_current_training_target_level'):
            self.player.set_current_training_target_level(target_level_idx)

    # Get observation
    def get_obs(self):
        # Render the game screen to the AI's observation canvas
        self.render_to_scaled_canvas(self.observation_canvas)

        # Visualize the charge value
        charge_value = self.player.jumpTimer
        if charge_value > 0:
            max_bar_width = int(IMAGE_INPUT_W * CHARGE_BAR_WIDTH_RATIO)
            bar_width = int((charge_value / MAX_JUMP_TIMER) * max_bar_width)
            
            bar_x = (IMAGE_INPUT_W - bar_width) // 2
            bar_y = IMAGE_INPUT_H - CHARGE_BAR_HEIGHT - CHARGE_BAR_Y_OFFSET

            pygame.draw.rect(self.observation_canvas, CHARGE_BAR_COLOR, (bar_x, bar_y, bar_width, CHARGE_BAR_HEIGHT))

        # Convert the surface to a Numpy array
        frame_rgb = pygame.surfarray.array3d(self.observation_canvas)

        # Transpose from (W, H, C) to (H, W, C)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        
        return frame_rgb.astype(np.uint8)

    # Reward function
    def calculate_reward(self, prev_info, current_info, done, landed_on_new_platform):
        reward = -0.02 # Time penalty

        current_height = current_info['global_height']
        prev_height = prev_info['global_height']
        height_diff = current_height - prev_height

        reward += height_diff * 0.02 # Reward/penalty for height change (per pixel)

        if landed_on_new_platform and len(self.visited_platforms) > 1: 
            reward += 2.5 # Reward for exploring a new platform

        if done:
            if current_info.get('stable_on_new_level_up', False): 
                reward += 20.0 # Reward for reaching the next level
            elif current_info.get('felt_to_previous_level', False):
                reward -= 5.0 # Penalty for falling to the previous level
            elif current_info.get('players_dead', False):
                reward -= 25.0 # Penalty for dying
        return np.clip(reward, -25.0, 25.0) # Clip the reward to a specific range

    # Execute a step in the environment
    def step(self, action):

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
        
        prev_global_height = self.player.getGlobalHeight()
        
        # Information from the previous step
        prev_info = {'global_height': prev_global_height, # height
                     'is_on_ground': self.player.isOnGround, # is on ground
                     'hasBumped': self.player.hasBumped # has bumped a wall
                     }

        # Update the game environment
        self.player.Update(single_mode=False)

        # Check for discovery of new platforms
        landed_on_new_platform = False
        if self.player.last_landed_line_id is not None:
            if self.player.last_landed_line_id not in self.visited_platforms:
                landed_on_new_platform = True
                self.visited_platforms.add(self.player.last_landed_line_id)

        current_global_height = self.player.getGlobalHeight()
        
        # Information from the current step
        current_info = {
            'global_height': current_global_height,
            'is_on_ground': self.player.isOnGround, 
            'stable_on_new_level_up': self.player.stable_on_new_level_up, # reached next level
            'players_dead': self.player.players_dead, 
            'felt_to_previous_level': self.player.felt_to_previous_level, # fell to previous level
            'player_x': self.player.x, 'player_y': self.player.y, 
            'hasBumped': self.player.hasBumped
        }
        
        # Failure condition
        failed = self.player.currentLevelNo < self.player.current_training_target_level

        # Termination condition
        terminated = self.player.players_dead or self.player.stable_on_new_level_up or failed
        
        # Current step count in the episode
        self.current_episode_step_count += 1

        # Truncation condition
        truncated = self.current_episode_step_count >= self._max_episode_steps
        done = terminated or truncated

        # Calculate the reward for this step
        reward = self.calculate_reward(prev_info, current_info, done, landed_on_new_platform)

        # Accumulate reward
        self.episode_total_reward += reward

        # Get observation
        observation = self.get_obs()
        
        # Information for logging
        info = {
            'global_height': self.player.getGlobalHeight(),
        }

        # Update the display
        if self.render_mode == "human": 
            self.render()
            
        return observation, reward, terminated, truncated, info

    # Reset the environment
    def reset(self, seed=None, options=None):
        # Initialization
        super().reset(seed=seed)
        self.current_episode_step_count = 0
        self.episode_total_reward = 0.0
        self.visited_platforms.clear()
        
        start_level_override = self.initial_target_level
        if options and "start_level" in options: 
            start_level_override = options["start_level"]
        
        # Ensure the level is within a valid range
        start_level_override = max(0, min(start_level_override, len(PREDEFINED_SPAWN_POINTS) - 1))
        
        # Reset to the training level
        if hasattr(self.player, 'set_current_training_target_level'): 
            self.player.set_current_training_target_level(start_level_override) 
        
        # Reset the player
        self.player.resetPlayer(initial_target_level=start_level_override)
        
        # Set spawn point with a bit of randomness
        if 0 <= start_level_override < len(PREDEFINED_SPAWN_POINTS):
            spawn_x, spawn_y = PREDEFINED_SPAWN_POINTS[start_level_override]
            random_offset = (np.random.rand() - 0.5) * 10
            self.player.x = spawn_x + random_offset
            self.player.y = spawn_y
        
        # Save a frame for debugging
        self.player.jumpTimer = 18 # Set charge to half value (temporary)
        observation = self.get_obs()
        self.player.jumpTimer = 0 # Reset immediately to avoid affecting training

        info = {'global_height': self.player.getGlobalHeight()}
        if self.render_mode == "human": self.render()

        return observation, info

    # Render the image to the small canvas
    def render_to_scaled_canvas(self, canvas):
        # Set the level
        level_to_draw = self.levels[self.player.currentLevelNo] if 0 <= self.player.currentLevelNo < len(self.levels) else None
        if level_to_draw:
            bg_image = level_to_draw.get_scaled_background()
            canvas.blit(bg_image, (0, 0))
        else:
            canvas.fill((50, 50, 50))
        img_to_draw, blit_pos = self.player.get_scaled_draw_info()
        if img_to_draw and blit_pos:
            canvas.blit(img_to_draw, blit_pos)

    # Render the game screen (for debugging)
    def render(self):
        if self.render_mode == "human":
            if self.window is None: return
            
            # Set the level
            level_to_draw = self.levels[self.player.currentLevelNo] if 0 <= self.player.currentLevelNo < len(self.levels) else None
            if level_to_draw:
                self.window.blit(level_to_draw.get_background(), (0, 0)) # Draw background
            else:
                self.window.fill((50, 50, 50))
            self.player.Draw(self.window, single_mode=True) # Draw player

            if self.font:
                # Debug display for level, height, and reward
                info_text = f"Lvl: {self.player.currentLevelNo + 1} H: {self.player.getGlobalHeight():.0f} R: {self.episode_total_reward:.2f}" 
                text_surface = self.font.render(info_text, True, (255, 255, 255))
                self.window.blit(text_surface, (10, 10))

            pygame.event.pump() # Pump events
            pygame.display.flip() # Flip the display

            self.clock.tick(self.metadata['render_fps'])

    # Close the environment
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()
