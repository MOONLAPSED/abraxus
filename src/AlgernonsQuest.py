#!/usr/bin/env python3
"""
Algernon's Quest - The Most Compressed Game Engine in History
A 3D maze game where everything is a ByteWord

Game Concept: Navigate a procedural maze where:
- World tiles are ByteWords
- Player actions are ByteWord compositions  
- Game logic is your morphic algebra
- Rendering is ByteWord pattern interpretation
"""

import time
import random
# --- TM Constants, Tables, Encoding/Decoding ---
# These are needed by the ByteWord.compose method, even if the exploration focuses
# on the general composition rule. They define the specific TM ontology embedded.

# Map TM states to specific ByteWord values (C=1 for states/verbs)
# Using distinct values that are unlikely to be produced by simple bitwise ops
BW_Q_FIND_1 = 0b10000000 # C=1, V=000, T=0000
BW_Q_HALT   = 0b11111111 # C=1, V=111, T=1111 (Distinct halt state)

# Map TM symbols to specific ByteWord values (C=0 for symbols/nouns)
BW_0_VAL        = 0b00000000 # C=0, __C=0, VV=00, T=0000
BW_1_VAL        = 0b00000001 # C=0, __C=0, VV=00, T=0001
BW_BLANK_VAL    = 0b00000101 # C=0, __C=0, VV=01, T=0101
BW_H_VAL        = 0b00001000 # C=0, __C=0, VV=01, T=1000 (Halt symbol)


# Dictionaries for easy lookup by value
TM_STATE_VALUES = {
    BW_Q_FIND_1,
    BW_Q_HALT,
}

TM_SYMBOL_VALUES = {
    BW_0_VAL,
    BW_1_VAL,
    BW_BLANK_VAL,
    BW_H_VAL,
}

# --- TM Transition Table ---
# (CurrentStateValue, SymbolReadValue) -> (NextStateValue, SymbolToWriteValue, MovementCode)
# Movement Codes: 00=Left, 01=Right, 10=Stay
MOVE_LEFT = 0b00
MOVE_RIGHT = 0b01
MOVE_STAY = 0b10

TM_TRANSITIONS = {
    (BW_Q_FIND_1, BW_0_VAL):     (BW_Q_FIND_1, BW_0_VAL, MOVE_RIGHT),
    (BW_Q_FIND_1, BW_1_VAL):     (BW_Q_HALT,   BW_H_VAL, MOVE_STAY),
    (BW_Q_FIND_1, BW_BLANK_VAL): (BW_Q_FIND_1, BW_BLANK_VAL, MOVE_RIGHT),
    (BW_Q_FIND_1, BW_H_VAL):     (BW_Q_FIND_1, BW_H_VAL, MOVE_RIGHT), # Handle encountering 'H' early
}

# --- Encoding/Decoding TM Outputs in Result ByteWord ---
# Encoding Scheme:
# Bits 0-1: Next State Index (00=Q_FIND_1, 01=Q_HALT) - Need a mapping for this
# Bits 2-3: Symbol to Write Index (00=0, 01=1, 10=BLANK, 11=H) - Need a mapping for this
# Bits 4-5: Head Movement (00=Left, 01=Right, 10=Stay)
# Bits 6-7: Always 00 (unused in this simple TM output encoding)

# Mapping state values to indices for encoding
STATE_TO_INDEX = {
    BW_Q_FIND_1: 0b00,
    BW_Q_HALT:   0b01,
}

# Mapping symbol values to indices for encoding
SYMBOL_TO_INDEX = {
    BW_0_VAL:     0b00,
    BW_1_VAL:     0b01,
    BW_BLANK_VAL: 0b10,
    BW_H_VAL:     0b11,
}

# Mapping indices back to state/symbol values for decoding
INDEX_TO_STATE = {v: k for k, v in STATE_TO_INDEX.items()}
INDEX_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_INDEX.items()}


def encode_tm_output(next_state_val, symbol_to_write_val, movement_code):
    """Encodes TM transition outputs into an 8-bit integer."""
    next_state_idx = STATE_TO_INDEX[next_state_val]
    symbol_idx = SYMBOL_TO_INDEX[symbol_to_write_val]

    encoded_value = (movement_code << 4) | (symbol_idx << 2) | next_state_idx
    # Ensure bits 6-7 are 0
    return encoded_value & 0b00111111

# Note: decode_tm_output is not strictly needed for this exploration script,
# but included for completeness if you were to run the TM simulation loop.
# def decode_tm_output(encoded_value):
#     """Decodes an 8-bit integer from compose into TM transition outputs."""
#     relevant_bits = encoded_value & 0b00111111
#     movement_code = (relevant_bits >> 4) & 0b11
#     symbol_idx = (relevant_bits >> 2) & 0b11
#     next_state_idx = relevant_bits & 0b11
#     next_state_val = INDEX_TO_STATE.get(next_state_idx)
#     symbol_to_write_val = INDEX_TO_SYMBOL.get(symbol_idx)
#     if next_state_val is None or symbol_to_write_val is None:
#          raise ValueError(f"Failed to decode TM output from value {encoded_value}: Invalid state or symbol index.")
#     return next_state_val, symbol_to_write_val, movement_code


# --- ByteWord Class Definition ---
class ByteWord:
    def __init__(self, value=0):
        # Ensure value is an 8-bit integer
        self._value = value & 0xFF

    # Properties for morphological components
    @property
    def C(self):
        return (self._value >> 7) & 0x01

    @property
    def _C(self):
        # Only relevant if C is 0
        return (self._value >> 6) & 0x01 if self.C == 0 else None

    @property
    def VV(self):
        # Only relevant if C is 0 (2 bits)
        return (self._value >> 4) & 0x03 if self.C == 0 else None

    @property
    def V(self):
        # Only relevant if C is 1 (3 bits)
        return (self._value >> 4) & 0x07 if self.C == 1 else None

    @property
    def T(self):
        # Relevant for both (4 bits)
        return self._value & 0x0F

    # Check if this ByteWord is a TM State (by value)
    def is_tm_state(self):
        return self._value in TM_STATE_VALUES

    # Check if this ByteWord is a TM Symbol (by value)
    def is_tm_symbol(self):
        return self._value in TM_SYMBOL_VALUES

    # General ByteWord composition rule (used when not a TM transition)
    def _general_compose(self, other):
        # Example non-associative bitwise rule (can be replaced)
        # This is different from the JS example to show flexibility
        # Let's use a simple mix: result_bits = (self_bits rotated) XOR (other_bits)
        self_rotated = ((self._value << 3) | (self._value >> 5)) & 0xFF # Rotate left by 3 bits
        result_value = self_rotated ^ other._value
        return ByteWord(result_value)

    # The main composition method - acts as the universal interaction law
    def compose(self, other):
        # --- TM Transition Logic ---
        # Check if self is a TM State and other is a TM Symbol
        if self.is_tm_state() and other.is_tm_symbol():
            transition_key = (self._value, other._value)
            if transition_key in TM_TRANSITIONS:
                # Found a TM transition rule
                next_state_val, symbol_to_write_val, movement_code = TM_TRANSITIONS[transition_key]
                # Encode the TM outputs into the result ByteWord's value
                encoded_result_value = encode_tm_output(next_state_val, symbol_to_write_val, movement_code)
                # The result ByteWord's value *is* the encoded TM instruction
                return ByteWord(encoded_result_value)
            else:
                # No specific TM rule for this State-Symbol pair defined
                # Fall back to general composition
                # print(f"DEBUG: No specific TM rule for {self} * {other}. Using general compose.") # Optional debug
                pass # Fall through to general compose

        # If not a TM State * TM Symbol pair, or no specific TM rule found, use the general rule
        return self._general_compose(other)

    def __str__(self):
        hex_val = hex(self._value)[2:].zfill(2).upper()
        binary_val = bin(self._value)[2:].zfill(8)

        morphology = f"C:{self.C}"
        if self.C == 0:
            morphology += f", _C:{self._C}, VV:{bin(self.VV)[2:].zfill(2)}"
        else:
            morphology += f", V:{bin(self.V)[2:].zfill(3)}"
        morphology += f", T:{bin(self.T)[2:].zfill(4)}"

        return f"ByteWord({morphology}) [0x{hex_val}, 0b{binary_val}]"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, ByteWord):
            return self._value == other._value
        return False

    def __hash__(self):
        return hash(self._value)

    def __format__(self, format_spec):
        return format(str(self), format_spec)


# --- Helper Functions to Create ByteWord from Morphology ---
def create_verb_from_morphology(v, t):
    """
    Creates a C=1 (Verb) ByteWord from specified morphological bits.
    v: Value for V (3 bits). Pass as integer (0-7).
    t: Value for T (4 bits). Pass as integer (0-15).
    """
    if not (0 <= v <= 7) or not (0 <= t <= 15):
         raise ValueError("Invalid V or T value for C=1 morphology")
    c = 1
    value = (c << 7) | (v << 4) | t
    return ByteWord(value)

def create_noun_from_morphology(cc, vv, t):
    """
    Creates a C=0 (Noun) ByteWord from specified morphological bits.
    cc: Value for _C (1 bit). Pass as integer (0 or 1).
    vv: Value for VV (2 bits). Pass as integer (0-3).
    t: Value for T (4 bits). Pass as integer (0-15).
    """
    if not (0 <= cc <= 1) or not (0 <= vv <= 3) or not (0 <= t <= 15):
         raise ValueError("Invalid _C, VV, or T value for C=0 morphology")
    c = 0
    value = (c << 7) | (cc << 6) | (vv << 4) | t
    return ByteWord(value)


# --- Exploration Function ---
def explore_operator_ket_t_effect(operator_template, ket_template_prefix_value):
    """
    Explores the effect of an operator template on kets varying only in their T field.

    Args:
        operator_template: A ByteWord instance representing the first operand template.
        ket_template_prefix_value: An 8-bit integer value for the ket,
                                   where the last 4 bits (T field) will be varied from 0 to 15.
                                   The C, _C/VV/V bits of the ket are determined by this prefix.
    """
    print(f"\n--- Exploring Operator: {operator_template} ---")
    # Determine ket morphology prefix string for printing
    temp_ket_prefix_bw = ByteWord(ket_template_prefix_value & 0b11110000)
    ket_prefix_str = f"C:{temp_ket_prefix_bw.C}"
    if temp_ket_prefix_bw.C == 0:
         ket_prefix_str += f", _C:{temp_ket_prefix_bw._C}, VV:{bin(temp_ket_prefix_bw.VV)[2:].zfill(2)}"
    else:
         ket_prefix_str += f", V:{bin(temp_ket_prefix_bw.V)[2:].zfill(3)}"

    print(f"--- Ket Template Prefix: [{ket_prefix_str}, T:xxxx] ---")
    print("-" * 80) # Adjusted width for better formatting

    # Header for the table
    print(f"{'Ket T (bin)':<15} | {'Ket ByteWord':<40} | {'Result ByteWord (General Compose)':<40}")
    print("-" * 15 + "-|-" + "-" * 40 + "-|-" + "-" * 40)

    for t_value in range(16):
        # Construct the ket ByteWord by setting the T bits
        ket_value = (ket_template_prefix_value & 0b11110000) | t_value
        ket_bw = ByteWord(ket_value)

        # Perform the composition using the general rule
        # We call _general_compose directly to isolate this rule's effect,
        # ignoring the TM logic branch for this specific exploration view.
        result_bw = operator_template._general_compose(ket_bw)

        # Print the interaction
        print(f"{bin(t_value)[2:].zfill(4):<15} | {ket_bw:<40} | {result_bw:<40}")

    print("-" * 80)


class AlgernonEngine:
    """The entire game engine in one class"""
    
    def __init__(self, width=16, height=16):
        self.width = width
        self.height = height
        
        # World is just a 2D array of ByteWords
        self.world = self._generate_maze()
        
        # Player state is a single ByteWord
        self.player = ByteWord(0b10010001)  # C=1, V=001, T=0001
        self.player_x = 1
        self.player_y = 1
        
        # Direction ByteWords for movement
        self.NORTH = ByteWord(0b10000001)   # C=1, V=000, T=0001  
        self.SOUTH = ByteWord(0b10000010)   # C=1, V=000, T=0010
        self.EAST = ByteWord(0b10000100)    # C=1, V=000, T=0100
        self.WEST = ByteWord(0b10001000)    # C=1, V=000, T=1000
        
        # World tile types
        self.WALL = ByteWord(0b11110000)    # C=1, V=111, T=0000
        self.FLOOR = ByteWord(0b00000000)   # C=0, all zeros
        self.EXIT = ByteWord(0b01010101)    # C=0, special pattern
        
        # Rendering characters for each ByteWord pattern
        self.render_map = self._build_render_map()
        
    def _generate_maze(self):
        """Generate a ByteWord maze using morphic patterns"""
        maze = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if x == 0 or x == self.width-1 or y == 0 or y == self.height-1:
                    # Border walls
                    cell = ByteWord(0b11110000)
                elif (x + y) % 3 == 0:
                    # Internal walls based on morphic pattern
                    cell = ByteWord(0b11110000) 
                elif x == self.width-2 and y == self.height-2:
                    # Exit
                    cell = ByteWord(0b01010101)
                else:
                    # Floor with slight variations
                    variation = (x * y) % 16
                    cell = ByteWord(0b00000000 | variation)
                row.append(cell)
            maze.append(row)
        return maze
    
    def _build_render_map(self):
        """Map ByteWord patterns to ASCII characters"""
        render_map = {}
        
        # Walls (C=1, V=111)
        for t in range(16):
            wall_byte = ByteWord(0b11110000 | t)
            render_map[wall_byte._value] = '#'
            
        # Floors (C=0)  
        for t in range(16):
            floor_byte = ByteWord(0b00000000 | t)
            render_map[floor_byte._value] = '.'
            
        # Special cases
        render_map[0b01010101] = 'E'  # Exit
        
        return render_map
    
    def move_player(self, direction_byte):
        """Move player using ByteWord composition"""
        
        # Get current world cell
        current_cell = self.world[self.player_y][self.player_x]
        
        # Compose player + direction + world_cell for movement decision
        movement_result = self.player.compose(direction_byte).compose(current_cell)
        
        # Extract movement from composed ByteWord
        # Use the T field to determine if movement is allowed
        movement_allowed = (movement_result.T & 0x01) == 1
        
        if movement_allowed:
            # Calculate new position based on direction
            new_x, new_y = self._get_new_position(direction_byte)
            
            # Check bounds and wall collision using ByteWord properties
            if (0 <= new_x < self.width and 0 <= new_y < self.height):
                target_cell = self.world[new_y][new_x]
                
                # Wall check: C=1 and V=111 means solid wall
                if not (target_cell.C == 1 and target_cell.V == 0b111):
                    self.player_x = new_x
                    self.player_y = new_y
                    
                    # Update player state based on new environment
                    self.player = self.player.compose(target_cell)
                    
                    # Check for exit condition
                    if target_cell._value == 0b01010101:
                        return "WIN"
        
        return "CONTINUE"
    
    def _get_new_position(self, direction_byte):
        """Convert direction ByteWord to coordinate delta"""
        # Use T field pattern to determine direction
        t_val = direction_byte.T
        
        if t_val == 0b0001:    # NORTH
            return self.player_x, self.player_y - 1
        elif t_val == 0b0010:  # SOUTH  
            return self.player_x, self.player_y + 1
        elif t_val == 0b0100:  # EAST
            return self.player_x + 1, self.player_y
        elif t_val == 0b1000:  # WEST
            return self.player_x - 1, self.player_y
        else:
            return self.player_x, self.player_y
    
    def render_frame(self):
        """Render the current game state as ASCII"""
        frame = []
        
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if x == self.player_x and y == self.player_y:
                    row += '@'  # Player character
                else:
                    cell_value = self.world[y][x]._value
                    char = self.render_map.get(cell_value, '?')
                    row += char
            frame.append(row)
        
        return '\n'.join(frame)
    
    def get_status(self):
        """Get game status as ByteWord values"""
        return {
            'player_byte': f"0x{self.player._value:02x}",
            'player_morphology': f"C:{self.player.C}, V:{self.player.V}, T:{self.player.T}",
            'position': f"({self.player_x}, {self.player_y})",
            'current_cell': f"0x{self.world[self.player_y][self.player_x]._value:02x}"
        }

def play_algernons_quest():
    """Main game loop"""
    game = AlgernonEngine()
    
    print("=== ALGERNON'S QUEST ===")
    print("Navigate the ByteWord maze to reach the exit (E)")
    print("Controls: WASD to move, Q to quit")
    print("Everything you see is a ByteWord with morphic properties!")
    print()
    
    while True:
        # Clear screen (simple version)
        print("\n" * 50)
        
        # Render game
        print(game.render_frame())
        print()
        
        # Show status
        status = game.get_status()
        print(f"Player: {status['player_byte']} {status['player_morphology']}")
        print(f"Position: {status['position']}")
        print(f"Standing on: {status['current_cell']}")
        print()
        
        # Get input
        move = input("Move (WASD) or Q to quit: ").lower()
        
        if move == 'q':
            break
        elif move == 'w':
            result = game.move_player(game.NORTH)
        elif move == 's':
            result = game.move_player(game.SOUTH)
        elif move == 'd':
            result = game.move_player(game.EAST)
        elif move == 'a':
            result = game.move_player(game.WEST)
        else:
            continue
            
        if result == "WIN":
            print("\n🎉 CONGRATULATIONS! You've mastered the ByteWord maze!")
            print("You are now a certified Morphic Navigator!")
            break

# For web compilation target
def compile_to_javascript():
    """Generate a JS version of the game"""
    js_code = """
// Algernon's Quest - JavaScript ByteWord Engine
class ByteWord {
    constructor(value) { this.value = value & 0xFF; }
    get C() { return (this.value >> 7) & 1; }
    get V() { return (this.value >> 4) & 0x07; }
    get T() { return this.value & 0x0F; }
    
    compose(other) {
        // Simplified composition for web
        let result = ((this.value << 3) | (this.value >> 5)) & 0xFF;
        return new ByteWord(result ^ other.value);
    }
}

// ... rest of game engine in JavaScript
"""
    return js_code

if __name__ == "__main__":
    # Uncomment to play the game
    play_algernons_quest()
    
    # Or compile to web
    print("ByteWord Game Engine Ready!")
    print("This is LISP with 8-bit words instead of S-expressions!")
    print("And it's also a game engine!")
    print("And it compiles to web!")
    print("🤯")