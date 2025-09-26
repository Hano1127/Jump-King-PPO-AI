import pygame

from vector import Vector, VectorNormalize, VectorDot, VectorMult 
from levelSetupFunction import MAP_LINES 
import settings
from line import Point as LinePoint, DiagonalCollisionInfo, AreLinesColliding

def mapNumber(n, a1, b1, a2, b2):
    if b1 - a1 == 0: return a2 
    return ((n-a1)*((b2-a2) / (b1-a1))) + a2

def numberInRange(a, b1, b2): 
    return (b1 <= a <= b2) or (b2 <= a <= b1)

class Player:
    runCycleIdx = 0
    runCycle = []
    scaledRunCycleIdx = 0 
    scaledRunCycle = [] 

    @classmethod
    def load_sprites(cls):
        if not cls.runCycle:
            if all(img is not None for img in [settings.PLAYER_RUN_IMAGE_1, settings.PLAYER_RUN_IMAGE_2, settings.PLAYER_RUN_IMAGE_3]):
                cls.runCycle = (
                    [settings.PLAYER_RUN_IMAGE_1] * 13 +
                    [settings.PLAYER_RUN_IMAGE_2] * 6 +
                    [settings.PLAYER_RUN_IMAGE_3] * 12 +
                    [settings.PLAYER_RUN_IMAGE_2] * 6
                )
                cls.scaledRunCycle = (
                    [settings.SCALED_PLAYER_RUN_IMAGE_1] * 13 +
                    [settings.SCALED_PLAYER_RUN_IMAGE_2] * 6 +
                    [settings.SCALED_PLAYER_RUN_IMAGE_3] * 12 +
                    [settings.SCALED_PLAYER_RUN_IMAGE_2] * 6
                )

    def __init__(self, initial_target_level=0):
        self.x = settings.WIDTH/2; self.y = settings.HEIGHT/2 
        self.w = settings.PLAYER_WIDTH; self.h = settings.PLAYER_HEIGHT 
        self.velx = 0.0; self.vely = 0.0
        self.pvelx = 0.0; self.pvely = 0.0 
        self.jumpTimer = 0
        self.jumpStartHeight = 0 
        self.isOnGround = False 
        self.isRunning = False
        self.isSliding = False 
        self.isSlidingLeft = False
        self.hasBumped = False
        self.hasFallen = False
        self.facingRight = True
        self.jumpHeld = False; self.leftHeld = False; self.rightHeld = False
        self.jump_released = False 
        self.max_collisions_checks = 20
        self.current_number_of_collision_checks = 0 
        self.players_dead = False
        self.best_level_reached = 0
        self.felt_to_previous_level = False 
        self.stable_on_new_level_up = False 
        self.currentLevelNo = initial_target_level
        self.current_training_target_level = initial_target_level 
        
        self.last_landed_line_id = None

    def set_current_training_target_level(self, new_target_level):
        self.current_training_target_level = new_target_level

    def resetPlayer(self, initial_target_level=0):
        self.x = settings.WIDTH / 2
        self.y = settings.HEIGHT / 2
        self.velx = 0.0; self.vely = 0.0
        self.pvelx = 0.0; self.pvely = 0.0
        self.isOnGround = False
        self.jumpTimer = 0
        self.jumpHeld = False; self.leftHeld = False; self.rightHeld = False
        self.jump_released = False
        self.isRunning = False; self.isSliding = False; self.isSlidingLeft = False
        self.hasBumped = False; self.facingRight = True; self.hasFallen = False
        self.jumpStartHeight = 0
        self.current_number_of_collision_checks = 0
        self.players_dead = False
        self.felt_to_previous_level = False
        self.currentLevelNo = initial_target_level
        self.current_training_target_level = initial_target_level
        self.stable_on_new_level_up = False
        
        self.last_landed_line_id = None

    def getGlobalHeight(self): 
        return (settings.HEIGHT - self.y) + settings.HEIGHT * self.currentLevelNo

    def isPlayerOnGround(self, currentLines): 
        self.y += 1
        for line in currentLines:
            if line.isHorizontal and self.CheckCollideWithLine(line):
                self.y -= 1
                return True
        self.y -= 1
        return False

    def isPlayerOnDiagonal(self, currentLines): 
        self.y += 5 
        for line in currentLines:
            if line.isDiagonal and self.CheckCollideWithLine(line):
                self.y -= 5
                return True
        self.y -= 5
        return False

    def isMovingLeft(self): return self.velx < 0 
    def isMovingRight(self): return self.velx > 0 
    def isMovingUp(self): return self.vely < 0 
    def isMovingDown(self): return self.vely > 0 

    def CheckCollideWithLine(self, line):
        if line.isHorizontal:
            isRectWithinLineX = (line.x1 < self.x and self.x < line.x2) or \
                                (line.x1 < self.x + self.w and self.x + self.w < line.x2) or \
                                (self.x < line.x1 and line.x1 < self.x + self.w) or \
                                (self.x < line.x2 and line.x2 < self.x + self.w)
            isRectWithinLineY = self.y < line.y1 and line.y1 < self.y + self.h
            return isRectWithinLineX and isRectWithinLineY
        elif line.isVertical:
            isRectWithinLineY = (line.y1 < self.y and self.y < line.y2) or \
                                (line.y1 < self.y + self.h and self.y + self.h < line.y2) or \
                                (self.y < line.y1 and line.y1 < self.y + self.h) or \
                                (self.y < line.y2 and line.y2 < self.y + self.h)
            isRectWithinLineX = self.x < line.x1 and line.x1 < self.x + self.w
            return isRectWithinLineX and isRectWithinLineY
        else:
            tl = (self.x, self.y)
            tr = (self.x + self.w, self.y)
            bl = (self.x, self.y + self.h - 1)
            br = (self.x + self.w, self.y + self.h - 1)
            
            leftCollision = AreLinesColliding(tl, bl, line.p1, line.p2)
            rightCollision = AreLinesColliding(tr, br, line.p1, line.p2)
            topCollision = AreLinesColliding(tl, tr, line.p1, line.p2)
            bottomCollision = AreLinesColliding(bl, br, line.p1, line.p2)
            
            if leftCollision[0] or rightCollision[0] or topCollision[0] or bottomCollision[0]:
                if not hasattr(line, 'diagonalCollisionInfo') or line.diagonalCollisionInfo is None:
                    line.diagonalCollisionInfo = DiagonalCollisionInfo()
                
                line.diagonalCollisionInfo.collidePlayerL = leftCollision[0]
                line.diagonalCollisionInfo.collidePlayerR = rightCollision[0]
                line.diagonalCollisionInfo.collidePlayerT = topCollision[0]
                line.diagonalCollisionInfo.collidePlayerB = bottomCollision[0]
                
                line.diagonalCollisionInfo.collisionPoints = []
                if leftCollision[0]: line.diagonalCollisionInfo.collisionPoints.append(LinePoint(leftCollision[1], leftCollision[2]))
                if rightCollision[0]: line.diagonalCollisionInfo.collisionPoints.append(LinePoint(rightCollision[1], rightCollision[2]))
                if topCollision[0]: line.diagonalCollisionInfo.collisionPoints.append(LinePoint(topCollision[1], topCollision[2]))
                if bottomCollision[0]: line.diagonalCollisionInfo.collisionPoints.append(LinePoint(bottomCollision[1], bottomCollision[2]))
                
                return True
            return False

    def GetPriorityCollision(self, collidedLines):
        if not collidedLines: return None
        if len(collidedLines) == 1: return collidedLines[0]

        if len(collidedLines) >= 2:
            vert = None; horiz = None
            for line in collidedLines:
                if line.isVertical: vert = line
                if line.isHorizontal: horiz = line
            
            if vert is not None and horiz is not None:
                prev_x = self.x - self.velx
                prev_y = self.y - self.vely
                time_to_collide_x = float('inf')
                if self.velx != 0:
                    dist_x = vert.x1 - (prev_x + self.w) if self.velx > 0 else vert.x1 - prev_x
                    time_to_collide_x = dist_x / self.velx
                
                time_to_collide_y = float('inf')
                if self.vely != 0:
                    dist_y = horiz.y1 - (prev_y + self.h) if self.vely > 0 else horiz.y1 - prev_y
                    time_to_collide_y = dist_y / self.vely

                if time_to_collide_x < 0: time_to_collide_x = float('inf')
                if time_to_collide_y < 0: time_to_collide_y = float('inf')

                if time_to_collide_x < time_to_collide_y - 1e-6: return vert
                if time_to_collide_y < time_to_collide_x - 1e-6: return horiz
                
                if self.isMovingUp() and self.velx != 0:
                    return vert
                
                return horiz
        
        min_correction_val = float('inf') 
        chosen_line = collidedLines[0]
        max_allowed_x_correction = -self.velx
        max_allowed_y_correction = -self.vely

        for line in collidedLines:
            calc_dc_x, calc_dc_y = 0, 0
            current_correction_val = float('inf')
            
            if line.isHorizontal:
                calc_dc_y = line.y1 - (self.y + self.h) if self.isMovingDown() else line.y1 - self.y
                current_correction_val = abs(calc_dc_y)
            elif line.isVertical:
                calc_dc_x = line.x1 - (self.x + self.w) if self.isMovingRight() else line.x1 - self.x
                current_correction_val = abs(calc_dc_x)

            if numberInRange(calc_dc_x, 0, max_allowed_x_correction) and \
               numberInRange(calc_dc_y, 0, max_allowed_y_correction):
                if current_correction_val < min_correction_val:
                    min_correction_val = current_correction_val
                    chosen_line = line
        return chosen_line

    def CheckCollisions(self, lines): 
        collided_lines = [line for line in lines if self.CheckCollideWithLine(line)]
        if not collided_lines: return

        chosen_line = self.GetPriorityCollision(collided_lines)
        if chosen_line is None: return

        potential_landing = False

        if chosen_line.isHorizontal:
            if self.isMovingDown():
                self.y = chosen_line.y1 - self.h
                if len(collided_lines) > 1:
                    potential_landing = True
                    self.velx = 0; self.vely = 0
                else: self.Land(chosen_line)
            else: 
                self.vely = -self.vely / 2
                self.y = chosen_line.y1
                self.hasBumped = True

        elif chosen_line.isVertical:
            if self.isMovingRight(): self.x = chosen_line.x1 - self.w
            elif self.isMovingLeft(): self.x = chosen_line.x1
            else:
                self.x = chosen_line.x1 - self.w if self.pvelx > 0 else chosen_line.x1
            self.velx = -self.velx / 2
            if not self.isOnGround: self.hasBumped = True
        
        else: 
            self.isSliding = True; self.hasBumped = True
            info = chosen_line.diagonalCollisionInfo
            if info and info.collisionPoints:
                if len(info.collisionPoints) == 2:
                    midpoint = LinePoint((info.collisionPoints[0].x + info.collisionPoints[1].x) / 2, 
                                         (info.collisionPoints[0].y + info.collisionPoints[1].y) / 2)
                    
                    player_corner_pos = None
                    if info.collidePlayerT and info.collidePlayerL: player_corner_pos = LinePoint(self.x, self.y)
                    elif info.collidePlayerT and info.collidePlayerR: player_corner_pos = LinePoint(self.x + self.w, self.y)
                    elif info.collidePlayerB and info.collidePlayerL: player_corner_pos = LinePoint(self.x, self.y + self.h); self.isSlidingLeft = False
                    elif info.collidePlayerB and info.collidePlayerR: player_corner_pos = LinePoint(self.x + self.w, self.y + self.h); self.isSlidingLeft = True
                    
                    if player_corner_pos:
                        self.x += midpoint.x - player_corner_pos.x
                        self.y += midpoint.y - player_corner_pos.y
                    
                    line_vec = VectorNormalize(Vector(chosen_line.x2 - chosen_line.x1, chosen_line.y2 - chosen_line.y1))
                    speed_vec = Vector(float(self.velx), float(self.vely))
                    speed_mag = VectorDot(speed_vec, line_vec)
                    new_vel = VectorMult(line_vec, speed_mag)
                    
                    self.velx, self.vely = new_vel
                    
                    if info.collidePlayerT: self.velx = 0; self.vely = 0; self.isSliding = False

                elif len(info.collisionPoints) == 1:
                    if info.collidePlayerL: 
                        self.x = max(chosen_line.x1, chosen_line.x2) + 1
                        self.velx = -self.velx/2 if self.isMovingLeft() else self.velx
                    elif info.collidePlayerR: 
                        self.x = min(chosen_line.x1, chosen_line.x2) - self.w - 1
                        self.velx = -self.velx/2 if self.isMovingRight() else self.velx
                    elif info.collidePlayerT: 
                        self.y = max(chosen_line.y1, chosen_line.y2) + 1
                        self.vely = -self.vely / 2
                    elif info.collidePlayerB: 
                        self.y = min(chosen_line.y1, chosen_line.y2) - self.h - 1
                        self.Land(chosen_line)


        if len(collided_lines) > 1:
            self.current_number_of_collision_checks += 1
            if self.current_number_of_collision_checks < self.max_collisions_checks:
                self.CheckCollisions(lines)
            if potential_landing:
                if self.isPlayerOnGround(lines):
                    self.Land(chosen_line)

    def CheckForLevelChange(self): 
        original_level_no = self.currentLevelNo
        if self.y < -self.h : self.currentLevelNo += 1; self.y += settings.HEIGHT 
        elif self.y > settings.HEIGHT:
            if self.currentLevelNo == 0: self.players_dead = True 
            else:
                self.currentLevelNo -= 1; self.y -= settings.HEIGHT
                if self.currentLevelNo < original_level_no: self.felt_to_previous_level = True
            if self.currentLevelNo < 0: self.players_dead = True; self.currentLevelNo = 0

    def ApplyGravity(self): 
        if self.isOnGround: self.vely = 0; return
        if self.isSliding: 
            self.vely = min(self.vely + settings.GRAVITY * 0.5, settings.TERMINAL_VELOCITY * 0.5) 
            if not self.isSlidingLeft: self.velx = min(self.velx + settings.GRAVITY * 0.5, settings.TERMINAL_VELOCITY * 0.5) 
            else: self.velx = max(self.velx - settings.GRAVITY * 0.5, -settings.TERMINAL_VELOCITY * 0.5) 
        else: self.vely = min(self.vely + settings.GRAVITY, settings.TERMINAL_VELOCITY) 

    def Jump(self): 
        if not self.isOnGround: return 
        jump_h_speed = settings.JUMP_SPEED_HORIZONTAL
        self.vely = -mapNumber(self.jumpTimer, 0, settings.MAX_JUMP_TIMER, settings.MIN_JUMP_SPEED, settings.MAX_JUMP_SPEED) 
        if self.leftHeld: self.velx = -jump_h_speed; self.facingRight = False
        elif self.rightHeld: self.velx = jump_h_speed; self.facingRight = True
        else: self.velx = 0
        self.hasFallen = False
        self.isOnGround = False
        self.jumpTimer = 0 
        self.jumpStartHeight = self.getGlobalHeight() 
        self.jump_released = True

    def Land(self, landed_line=None):
        self.isOnGround = True
        self.isSliding = False
        self.hasBumped = False
        if (self.jumpStartHeight - (settings.HEIGHT / 2) > self.getGlobalHeight()):
            self.hasFallen = True
        else:
            self.hasFallen = False
        
        if landed_line:
            self.last_landed_line_id = (self.currentLevelNo, landed_line.x1, landed_line.y1, landed_line.x2, landed_line.y2)

        self.velx = 0; self.vely = 0

    def UpdatePlayerSlide(self, lines): 
        if self.isSliding:
            if not self.isPlayerOnDiagonal(lines): self.isSliding = False 

    def UpdatePlayerRun(self, lines):
        self.isRunning = False
        if not self.isOnGround: return
        if not self.isPlayerOnGround(lines):
            self.isOnGround = False; return
        
        if self.jumpHeld and not self.jump_released:
            self.velx = 0
            self.isRunning = False
        else:
            if self.rightHeld:
                self.isRunning = True; self.facingRight = True
                self.velx = settings.RUN_SPEED
            elif self.leftHeld:
                self.isRunning = True; self.facingRight = False
                self.velx = -settings.RUN_SPEED
            else:
                self.isRunning = False
                self.velx = 0

    def UpdatePlayerJump(self):
        if not self.jumpHeld and self.jumpTimer > 0 and self.isOnGround: self.Jump()

    def UpdateJumpTimer(self):
        if self.isOnGround and self.jumpHeld:
            if not self.jump_released:
                self.jumpTimer = min(self.jumpTimer + 1, settings.MAX_JUMP_TIMER)
                if self.jumpTimer >= settings.MAX_JUMP_TIMER: self.Jump()
        elif not self.jumpHeld:
            self.jump_released = False

    def Update(self, single_mode=True): 
        if self.players_dead: return
        self.pvelx, self.pvely = self.velx, self.vely
        self.stable_on_new_level_up = False 
        self.felt_to_previous_level = False
        
        self.last_landed_line_id = None
        
        if self.hasFallen and (self.leftHeld or self.rightHeld or self.jumpHeld) and self.isOnGround:
            self.hasFallen = False

        current_map_lines = []
        if 0 <= self.currentLevelNo < len(MAP_LINES) and MAP_LINES[self.currentLevelNo]:
            current_map_lines = MAP_LINES[self.currentLevelNo].get_lines()
        
        self.UpdatePlayerSlide(current_map_lines) 
        self.ApplyGravity(); self.UpdatePlayerRun(current_map_lines) 
        self.x += self.velx; self.y += self.vely
        self.current_number_of_collision_checks = 0 
        self.CheckCollisions(current_map_lines) 
        self.UpdateJumpTimer(); self.UpdatePlayerJump() 
        self.CheckForLevelChange() 

        if self.isOnGround and self.currentLevelNo > self.current_training_target_level:
            self.stable_on_new_level_up = True
        
        if self.currentLevelNo > self.best_level_reached: self.best_level_reached = self.currentLevelNo
        

    def GetSpriteToDraw(self):
        if not Player.runCycle: return settings.PLAYER_IDLE_IMAGE
        if self.jumpHeld and self.isOnGround and not self.isSliding and not self.jump_released: return settings.PLAYER_SQUAT_IMAGE
        if self.hasFallen: return settings.PLAYER_FALLEN_IMAGE 
        if self.hasBumped: return settings.PLAYER_BUMP_IMAGE 
        if self.vely < 0 and not self.isOnGround: return settings.PLAYER_JUMP_IMAGE 
        if self.isRunning and self.isOnGround: 
            Player.runCycleIdx = (Player.runCycleIdx + 1) % len(Player.runCycle)
            return Player.runCycle[Player.runCycleIdx]
        if self.isOnGround: return settings.PLAYER_IDLE_IMAGE 
        return settings.PLAYER_FALL_IMAGE 

    def GetScaledSpriteToDraw(self):
        if not Player.scaledRunCycle: return settings.SCALED_PLAYER_IDLE_IMAGE
        if self.jumpHeld and self.isOnGround and not self.isSliding and not self.jump_released: return settings.SCALED_PLAYER_SQUAT_IMAGE
        if self.hasFallen: return settings.SCALED_PLAYER_FALLEN_IMAGE 
        if self.hasBumped: return settings.SCALED_PLAYER_BUMP_IMAGE 
        if self.vely < 0 and not self.isOnGround: return settings.SCALED_PLAYER_JUMP_IMAGE 
        if self.isRunning and self.isOnGround: 
            Player.scaledRunCycleIdx = (Player.scaledRunCycleIdx + 1) % len(Player.scaledRunCycle)
            return Player.scaledRunCycle[Player.scaledRunCycleIdx]
        if self.isOnGround: return settings.SCALED_PLAYER_IDLE_IMAGE 
        return settings.SCALED_PLAYER_FALL_IMAGE

    def get_draw_info(self):
        if self.players_dead:
            return None, None, None
        img = self.GetSpriteToDraw()
        if img is None: return None, None, None
        img_rect = img.get_rect()
        draw_x = self.x - 20 
        draw_y = self.y + self.h - img_rect.height
        if self.hasBumped or img in [settings.PLAYER_JUMP_IMAGE, settings.PLAYER_FALL_IMAGE]:
            draw_y += 5
        img_to_draw = img if self.facingRight else pygame.transform.flip(img, True, False)
        blit_pos = (int(draw_x), int(draw_y))
        if not self.facingRight:
            blit_pos = (int(self.x + self.w + 20 - img_rect.width), int(draw_y))
        img_rect.topleft = blit_pos
        return img_to_draw, blit_pos, img_rect

    def get_scaled_draw_info(self):
        img = self.GetScaledSpriteToDraw()
        if img is None: return None, None
        scale_w = settings.IMAGE_INPUT_W / settings.WIDTH
        scale_h = settings.IMAGE_INPUT_H / settings.HEIGHT
        scaled_x = self.x * scale_w
        scaled_y = self.y * scale_h
        scaled_h_player = self.h * scale_h
        img_rect = img.get_rect()
        draw_x_offset = -20 * scale_w
        draw_y_offset = 5 * scale_h
        draw_x = scaled_x + draw_x_offset
        draw_y = scaled_y + scaled_h_player - img_rect.height
        if self.hasBumped or img in [settings.SCALED_PLAYER_JUMP_IMAGE, settings.SCALED_PLAYER_FALL_IMAGE]:
            draw_y += draw_y_offset
        img_to_draw = img if self.facingRight else pygame.transform.flip(img, True, False)
        if not self.facingRight:
            scaled_w_player = self.w * scale_w
            draw_x = scaled_x + scaled_w_player - img_rect.width - draw_x_offset
        blit_pos = (int(draw_x), int(draw_y))
        return img_to_draw, blit_pos

    def Draw(self, window, single_mode=True): 
        img_to_draw, blit_pos, _ = self.get_draw_info()
        if img_to_draw is None: return
        window.blit(img_to_draw, blit_pos)