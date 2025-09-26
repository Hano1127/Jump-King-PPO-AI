import json
import os
import pygame
from settings import IMAGE_PATH, WIDTH, HEIGHT, IMAGE_INPUT_W, IMAGE_INPUT_H

class Level:
    def __init__(self, bg_path, lines_data):
        self.bg_path = bg_path
        self.lines_data = lines_data
        self.lines = None
        self.lines_np = None 
        self._bg_surface_cache = None
        self._scaled_bg_surface_cache = None 

    def get_background(self):
        if self._bg_surface_cache is None:
            try:
                raw_image = pygame.image.load(self.bg_path)
                self._bg_surface_cache = pygame.transform.scale(raw_image, (WIDTH, HEIGHT))
            except pygame.error:
                surface = pygame.Surface((WIDTH, HEIGHT))
                surface.fill((50, 50, 50))
                self._bg_surface_cache = surface
        return self._bg_surface_cache

    def get_scaled_background(self):
        if self._scaled_bg_surface_cache is None:
            original_bg = self.get_background()
            self._scaled_bg_surface_cache = pygame.transform.smoothscale(
                original_bg, (IMAGE_INPUT_W, IMAGE_INPUT_H)
            )
        return self._scaled_bg_surface_cache
        
    def get_lines(self):
        if self.lines is None:
            from line import Line
            self.lines = [Line(*lineData) for lineData in self.lines_data]
        return self.lines


class MapLoader:
    def __init__(self, path):
        self.path = path
        with open(path, "r") as f:
            self.data = json.load(f)

    def loadLevels(self):
        levels = []
        for idx_str, level_data in self.data.items():
            idx_int = int(idx_str)
            bg_path = os.path.join(IMAGE_PATH, "bg", f"{idx_str}.png")
            
            level = Level(
                bg_path=bg_path,
                lines_data=level_data["lines"]
            )
            while len(levels) <= idx_int: levels.append(None)
            levels[idx_int - 1] = level
        return levels

MAP_LINES = MapLoader(os.path.join(os.path.dirname(__file__), "assets/map.json")).loadLevels()
