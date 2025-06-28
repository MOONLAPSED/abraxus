#!/usr/bin/env python3
"""
ByteWord Wolfenstein - A shit-tier 3D-ish game using morphic card animation
Each frame is a ByteWord that encodes the complete game state
"""

import time
import random
from dataclasses import dataclass
from typing import List, Tuple

# Import your ByteWord system
class ByteWord:
    def __init__(self, value=0):
        self._value = value & 0xFF

    @property
    def C(self):
        return (self._value >> 7) & 0x01

    @property
    def V(self):
        return (self._value >> 4) & 0x07 if self.C == 1 else None

    @property
    def VV(self):
        return (self._value >> 4) & 0x03 if self.C == 0 else None

    @property
    def T(self):
        return self._value & 0x0F

    def compose(self, other):
        # Simple composition for game physics
        # This is where the magic happens - each frame composes with input
        result = (self._value ^ other._value) & 0xFF
        return ByteWord(result)

    def __str__(self):
        return f"ByteWord(0x{self._value:02x})"

@dataclass
class GameCard:
    """A game frame represented as a card with ByteWord state"""
    frame_id: str
    byte_state: ByteWord
    world_layout: List[str]  # ASCII representation
    
    def render_ascii(self) -> str:
        """Render this card as ASCII art"""
        player_x = self.byte_state.T % 4
        player_y = (self.byte_state.T >> 2) % 4
        
        # Copy the world layout
        lines = [list(line) for line in self.world_layout]
        
        # Place player
        if 0 <= player_y < len(lines) and 0 <= player_x < len(lines[player_y]):
            lines[player_y][player_x] = '@'
        
        return '\n'.join(''.join(line) for line in lines)

class MorphicWolfenstein:
    """The game engine - each tick generates a new card"""
    
    def __init__(self):
        self.current_frame = 0
        self.game_state = ByteWord(0b10000000)  # C=1 (active), starting state
        
        # Base world layouts (encoded in different cards)
        self.world_templates = [
            # Room 1
            [
                "################",
                "#..............#",
                "#..............#", 
                "#..............#",
                "#..............#",
                "#..............#",
                "#..............#",
                "################"
            ],
            # Room 2  
            [
                "################",
                "#......##......#",
                "#......##......#",
                "#......##......#", 
                "#......##......#",
                "#......##......#",
                "#......##......#",
                "################"
            ],
            # Room 3
            [
                "################",
                "#..#........#..#",
                "#..#........#..#",
                "#..............#",
                "#..............#", 
                "#..#........#..#",
                "#..#........#..#",
                "################"
            ]
        ]
    
    def process_input(self, key: str) -> ByteWord:
        """Convert keyboard input to ByteWord"""
        input_map = {
            'w': ByteWord(0b10000001),  # Move forward
            's': ByteWord(0b10000010),  # Move back  
            'a': ByteWord(0b10000100),  # Turn left
            'd': ByteWord(0b10001000),  # Turn right
            ' ': ByteWord(0b11111111),  # Action/shoot
        }
        return input_map.get(key, ByteWord(0b00000000))
    
    def tick(self, input_key: str = None) -> GameCard:
        """Generate the next frame card"""
        # Get input ByteWord
        input_bw = self.process_input(input_key) if input_key else ByteWord(0)
        
        # Compose current state with input (this is the game physics!)
        new_state = self.game_state.compose(input_bw)
        
        # Select world template based on state
        world_idx = (new_state.T >> 2) % len(self.world_templates)
        world_layout = self.world_templates[world_idx]
        
        # Create the frame card
        card_id = f"frame_{self.current_frame:06d}"
        card = GameCard(
            frame_id=card_id,
            byte_state=new_state,
            world_layout=world_layout
        )
        
        # Update game state
        self.game_state = new_state
        self.current_frame += 1
        
        return card
    
    def get_game_info(self) -> str:
        """Debug info about current game state"""
        state = self.game_state
        return f"""
Frame: {self.current_frame}
ByteWord State: {state}
C (Active): {state.C}
V (Mode): {state.V}
T (Position): {state.T:04b} (x={state.T % 4}, y={(state.T >> 2) % 4})
"""

def play_game():
    """Simple game loop - press keys to move around"""
    game = MorphicWolfenstein()
    
    print("ByteWord Wolfenstein - Morphic Card Animation Demo")
    print("Controls: W/A/S/D to move, SPACE for action, Q to quit")
    print("Each frame is a ByteWord that composes with your input!")
    print("-" * 60)
    
    # Simulate 30fps for a few seconds with random input
    for i in range(90):  # 3 seconds at 30fps
        # Simulate random input (in real version, this would be keyboard)
        inputs = ['w', 's', 'a', 'd', ' ', None]
        random_input = random.choice(inputs)
        
        # Generate next frame
        frame_card = game.tick(random_input)
        
        # Clear screen (simple terminal animation)
        print("\033[2J\033[H", end="")  # ANSI clear screen
        
        # Render the card
        print(f"=== ByteWord Wolfenstein - {frame_card.frame_id} ===")
        print(frame_card.render_ascii())
        print(game.get_game_info())
        
        if random_input:
            print(f"Input: {random_input} -> {game.process_input(random_input)}")
        
        # 30fps timing
        time.sleep(1/30)
    
    print("\n" + "="*60)
    print("Demo complete! Each frame was a morphic card with ByteWord state.")
    print("The entire game runs on card composition at 30fps!")

if __name__ == "__main__":
    play_game()