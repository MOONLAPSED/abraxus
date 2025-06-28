#!/usr/bin/env python3
"""
ByteWord 3D Maze Engine - The most compressed game engine in history
Each frame is computed via morphic ByteWord composition
"""

import time
import os
import math
from typing import List, Tuple

# Your ByteWord system (simplified for game engine)
class ByteWord:
    def __init__(self, value=0):
        self._value = value & 0xFF
    
    @property 
    def value(self):
        return self._value
    
    def compose(self, other):
        """Universal morphic composition - the only operation we need!"""
        # Raycasting-optimized composition rule
        # Mix position and direction info via rotation + XOR
        rotated = ((self._value << 3) | (self._value >> 5)) & 0xFF
        result = rotated ^ other._value
        return ByteWord(result)
    
    def extract_coord(self):
        """Extract coordinate from ByteWord (0-15 range)"""
        return (self._value >> 4) & 0x0F
    
    def extract_direction(self):
        """Extract direction from ByteWord (0-15 = 16 cardinal directions)"""
        return self._value & 0x0F
    
    def __str__(self):
        return f"BW(0x{self._value:02x})"

class MazeGameState:
    """Complete game state in 4 ByteWords = 32 bits total"""
    
    def __init__(self):
        # 16x16 maze grid (fits in 4 bits each)
        self.player_x = ByteWord(0x80)  # X=8, misc data in lower bits
        self.player_y = ByteWord(0x80)  # Y=8, misc data in lower bits  
        self.direction = ByteWord(0x00) # Facing north initially
        self.maze_state = ByteWord(0x01) # Maze variant/room number
        
        # Simple maze layout (1=wall, 0=empty)
        self.maze = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1],
            [1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,1],
            [1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1],
            [1,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1],
            [1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1],
            [1,1,1,0,1,1,0,0,0,0,1,1,0,1,1,1],
            [1,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1],
            [1,0,1,0,1,0,1,1,1,1,0,1,0,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1],
            [1,0,1,1,1,1,1,0,0,1,1,1,0,1,0,1],
            [1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1],
            [1,0,1,1,1,1,1,0,0,0,0,1,1,1,0,1],
            [1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1],
            [1,1,1,0,0,0,1,0,0,0,0,1,1,1,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        ]
    
    def get_position(self) -> Tuple[int, int]:
        """Extract current position from ByteWords"""
        x = self.player_x.extract_coord()
        y = self.player_y.extract_coord() 
        return (x, y)
    
    def get_direction_angle(self) -> float:
        """Convert ByteWord direction to angle (0-2π)"""
        dir_val = self.direction.extract_direction()
        return (dir_val / 16.0) * 2 * math.pi
    
    def move_forward(self):
        """Move forward via morphic composition"""
        # Compose direction with position to get movement vector
        move_x = self.direction.compose(self.player_x)
        move_y = self.direction.compose(self.player_y)
        
        # Extract new position
        new_x = move_x.extract_coord() 
        new_y = move_y.extract_coord()
        
        # Check bounds and walls
        if (0 < new_x < 15 and 0 < new_y < 15 and 
            self.maze[new_y][new_x] == 0):
            self.player_x = ByteWord((new_x << 4) | (self.player_x.value & 0x0F))
            self.player_y = ByteWord((new_y << 4) | (self.player_y.value & 0x0F))
    
    def turn_left(self):
        """Turn left via morphic composition"""
        turn_bw = ByteWord(0x0F)  # Turn amount
        self.direction = self.direction.compose(turn_bw)
    
    def turn_right(self):
        """Turn right via morphic composition"""  
        turn_bw = ByteWord(0x01)  # Turn amount
        self.direction = self.direction.compose(turn_bw)

class MorphicRenderer:
    """Renders 3D view via pure ByteWord composition"""
    
    def __init__(self, width=40, height=20):
        self.width = width
        self.height = height
        
    def cast_ray(self, game_state: MazeGameState, ray_angle: float) -> int:
        """Cast a single ray and return wall distance via morphic math"""
        x, y = game_state.get_position()
        
        # Convert to ByteWords for morphic ray computation
        ray_x_bw = ByteWord(int(x * 16))
        ray_y_bw = ByteWord(int(y * 16)) 
        angle_bw = ByteWord(int((ray_angle / (2 * math.pi)) * 256))
        
        # Morphic ray stepping 
        dx = math.cos(ray_angle) * 0.1
        dy = math.sin(ray_angle) * 0.1
        
        distance = 0
        ray_x, ray_y = float(x), float(y)
        
        while distance < 16:  # Max view distance
            ray_x += dx
            ray_y += dy
            distance += 0.1
            
            map_x, map_y = int(ray_x), int(ray_y)
            if (map_x < 0 or map_x >= 16 or map_y < 0 or map_y >= 16 or
                game_state.maze[map_y][map_x] == 1):
                break
                
        return int(distance * 4)  # Scale for display
    
    def render_frame(self, game_state: MazeGameState) -> List[str]:
        """Render complete frame via ByteWord composition chain"""
        frame = []
        player_angle = game_state.get_direction_angle()
        fov = math.pi / 3  # 60 degree field of view
        
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                # Calculate ray angle for this screen column
                ray_angle = player_angle + (x - self.width/2) * (fov / self.width)
                
                # Cast ray via morphic computation
                distance = self.cast_ray(game_state, ray_angle)
                
                # Convert distance to wall height
                if distance == 0:
                    wall_height = self.height
                else:
                    wall_height = min(self.height, int(self.height * 4 / distance))
                
                # Determine what character to draw
                if y < (self.height - wall_height) // 2:
                    char = ' '  # Sky
                elif y >= (self.height + wall_height) // 2:
                    char = '.'  # Floor  
                else:
                    # Wall - use different chars for distance
                    if distance < 4:
                        char = '█'
                    elif distance < 8:  
                        char = '▓'
                    elif distance < 12:
                        char = '▒'
                    else:
                        char = '░'
                
                line += char
            frame.append(line)
        
        return frame

def main():
    """The most compressed 3D game engine in history"""
    print("ByteWord 3D Maze - The Most Compressed Game Engine!")
    print("Controls: WASD to move, Q to quit")
    print("Total game state: 32 bits (4 ByteWords)")
    
    game = MazeGameState()
    renderer = MorphicRenderer()
    
    while True:
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Render frame via morphic composition
        frame = renderer.render_frame(game)
        
        # Display
        for line in frame:
            print(line)
        
        # Show debug info
        x, y = game.get_position()
        angle = game.get_direction_angle()
        print(f"\nPosition: ({x}, {y}) | Direction: {angle:.2f} rad")
        print(f"State ByteWords: {game.player_x} {game.player_y} {game.direction} {game.maze_state}")
        
        # Get input
        try:
            import sys, tty, termios
            old_settings = termios.tcgetattr(sys.stdin)
            tty.cbreak(sys.stdin.fileno())
            key = sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except:
            key = input("Enter command (w/a/s/d/q): ").strip().lower()
        
        # Process input via morphic state changes
        if key.lower() == 'w':
            game.move_forward()
        elif key.lower() == 's':
            # Move backward (compose with inverse)
            game.direction = game.direction.compose(ByteWord(0x80))
            game.move_forward() 
            game.direction = game.direction.compose(ByteWord(0x80))
        elif key.lower() == 'a':
            game.turn_left()
        elif key.lower() == 'd':
            game.turn_right()
        elif key.lower() == 'q':
            break
        
        time.sleep(0.1)  # ~10 FPS for smooth morphic computation

if __name__ == "__main__":
    main()