import pygame
import os

# Path settings 
current_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(current_dir, "assets", "images")
SAVE_PATH = os.path.join(current_dir, "models")
CHECKPOINT_PATH = os.path.join(current_dir, "checkpoints")
LOG_PATH = os.path.join(current_dir, "logs")

# Game settings
FPS = 60
WIDTH = 1200 
HEIGHT = 900 
PLAYER_WIDTH = 47
PLAYER_HEIGHT = 65
MAX_LEVELS = 26

# AI input related settings
IMAGE_INPUT_W = 128
IMAGE_INPUT_H = 96

USE_GRAYSCALE = True 
FRAME_STACK_SIZE = 4

CONTRAST_FACTOR = 2.5


# Player physics parameters
RUN_SPEED = 3.4
MIN_JUMP_SPEED = 1
MAX_JUMP_SPEED = 22
MAX_JUMP_TIMER = 36
JUMP_SPEED_HORIZONTAL = 8.5
TERMINAL_VELOCITY = 20 
GRAVITY = 0.6
ACTION_DECISION_INTERVAL = 6

# Player images
PLAYER_RUN_IMAGE_1 = None
PLAYER_RUN_IMAGE_2 = None
PLAYER_RUN_IMAGE_3 = None
PLAYER_SQUAT_IMAGE = None
PLAYER_FALLEN_IMAGE = None
PLAYER_BUMP_IMAGE = None
PLAYER_JUMP_IMAGE = None
PLAYER_IDLE_IMAGE = None
PLAYER_FALL_IMAGE = None

# Scaled player images
SCALED_PLAYER_RUN_IMAGE_1 = None
SCALED_PLAYER_RUN_IMAGE_2 = None
SCALED_PLAYER_RUN_IMAGE_3 = None
SCALED_PLAYER_SQUAT_IMAGE = None
SCALED_PLAYER_FALLEN_IMAGE = None
SCALED_PLAYER_BUMP_IMAGE = None
SCALED_PLAYER_JUMP_IMAGE = None
SCALED_PLAYER_IDLE_IMAGE = None
SCALED_PLAYER_FALL_IMAGE = None

# Load a single original size image
def load_player_image(filename, width, height):
    try:
        path = os.path.join(IMAGE_PATH, "poses", filename)
        image = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(image, (int(width), int(height)))
    except pygame.error as e:
        print(f"!!! Failed to load image: {filename}: {e}")
        placeholder = pygame.Surface((int(width), int(height)), pygame.SRCALPHA)
        placeholder.fill((255, 0, 255, 128))
        return placeholder

# Scale the given Surface to the observation size required by the AI
def scaled_player_image(original_image):
    if original_image is None:
        return None
    
    original_w, original_h = original_image.get_size()
    scale_factor = IMAGE_INPUT_H / HEIGHT
    scaled_w = int(original_w * scale_factor)
    scaled_h = int(original_h * scale_factor)
    
    if scaled_w <= 0 or scaled_h <= 0:
        return pygame.Surface((1, 1), pygame.SRCALPHA)
    
    scaled_img = pygame.transform.scale(original_image, (scaled_w, scaled_h))
    return scaled_img


# Load all images

_images_loaded = False
def load_all_images():
    global _images_loaded
    if _images_loaded:
        return
    
    if not pygame.get_init():
        pygame.init()

    global PLAYER_RUN_IMAGE_1, PLAYER_RUN_IMAGE_2, PLAYER_RUN_IMAGE_3
    global PLAYER_JUMP_IMAGE, PLAYER_IDLE_IMAGE, PLAYER_FALL_IMAGE
    global PLAYER_SQUAT_IMAGE, PLAYER_FALLEN_IMAGE, PLAYER_BUMP_IMAGE
    global SCALED_PLAYER_RUN_IMAGE_1, SCALED_PLAYER_RUN_IMAGE_2, SCALED_PLAYER_RUN_IMAGE_3
    global SCALED_PLAYER_JUMP_IMAGE, SCALED_PLAYER_IDLE_IMAGE, SCALED_PLAYER_FALL_IMAGE
    global SCALED_PLAYER_SQUAT_IMAGE, SCALED_PLAYER_FALLEN_IMAGE, SCALED_PLAYER_BUMP_IMAGE

    PLAYER_IDLE_IMAGE = load_player_image("idle.png", 93 * 0.9, 103 * 0.9)
    PLAYER_JUMP_IMAGE = load_player_image("jump.png", 93 * 0.9, 103 * 0.9)
    PLAYER_FALL_IMAGE = load_player_image("fall.png", 93 * 0.9, 103 * 0.9)
    PLAYER_RUN_IMAGE_1 = load_player_image("run1.png", 93 * 0.9, 103 * 0.9)
    PLAYER_RUN_IMAGE_2 = load_player_image("run2.png", 93 * 0.9, 103 * 0.9)
    PLAYER_RUN_IMAGE_3 = load_player_image("run3.png", 93 * 0.9, 103 * 0.9)
    PLAYER_SQUAT_IMAGE = load_player_image("squat.png", 93 * 0.9, 103 * 0.9)
    PLAYER_FALLEN_IMAGE = load_player_image("fallen.png", 93 * 0.9, 103 * 0.9)
    PLAYER_BUMP_IMAGE = load_player_image("bump.png", 93 * 0.9, 103 * 0.9)

    SCALED_PLAYER_IDLE_IMAGE = scaled_player_image(PLAYER_IDLE_IMAGE)
    SCALED_PLAYER_JUMP_IMAGE = scaled_player_image(PLAYER_JUMP_IMAGE)
    SCALED_PLAYER_FALL_IMAGE = scaled_player_image(PLAYER_FALL_IMAGE)
    SCALED_PLAYER_RUN_IMAGE_1 = scaled_player_image(PLAYER_RUN_IMAGE_1)
    SCALED_PLAYER_RUN_IMAGE_2 = scaled_player_image(PLAYER_RUN_IMAGE_2)
    SCALED_PLAYER_RUN_IMAGE_3 = scaled_player_image(PLAYER_RUN_IMAGE_3)
    SCALED_PLAYER_SQUAT_IMAGE = scaled_player_image(PLAYER_SQUAT_IMAGE)
    SCALED_PLAYER_FALLEN_IMAGE = scaled_player_image(PLAYER_FALLEN_IMAGE)
    SCALED_PLAYER_BUMP_IMAGE = scaled_player_image(PLAYER_BUMP_IMAGE)

    _images_loaded = True

# Spawn points
PREDEFINED_SPAWN_POINTS = [
    (590, 745), (884, 675), (568, 695), (480, 735), (340, 715), (540, 755), 
    (925, 795), (105, 795), (160, 675), (1000, 755), (895, 795), (900, 795), 
    (1120, 795), (620, 595), (300, 755), (90, 715), (1015, 715), (120, 675), 
    (980, 715), (340, 615), (615, 715), (640, 735), (345, 735), (630, 715), 
    (210, 695), (595, 715)
]

CHARGE_BAR_WIDTH_RATIO = 0.25      # Ratio of the charge bar's max width to the total screen width
CHARGE_BAR_HEIGHT = 2              # Height (thickness) of the charge bar
CHARGE_BAR_Y_OFFSET = 2            # Vertical distance of the charge bar from the top or bottom of the screen
CHARGE_BAR_COLOR = (0, 0, 0)       # Color of the charge bar (RGB)
