import sys # Included for completeness, though not strictly used in this snippet

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


# --- Examples of Exploration ---

# 1. Explore a C=1 (Verb) operator with V=000, T=0000
#    How does BW(C:1, V:000, T:0000) interact with kets varying only in T?
#    Let's use a generic C=0, __C=0, VV=00 ket template (0b0000xxxx)
operator_v0_t0 = create_verb_from_morphology(v=0b000, t=0b0000)
generic_noun_ket_prefix_value = 0b00000000 # C=0, __C=0, VV=00, T=xxxx
explore_operator_ket_t_effect(operator_v0_t0, generic_noun_ket_prefix_value)

# 2. Explore a C=1 (Verb) operator with V=111, T=1111
#    How does BW(C:1, V=111, T=1111) interact with the same generic ket?
operator_v7_t15 = create_verb_from_morphology(v=0b111, t=0b1111)
explore_operator_ket_t_effect(operator_v7_t15, generic_noun_ket_prefix_value)

# 3. Explore a C=0 (Noun) operator with __C=0, VV=00, T=0000
#    How does BW(C:0, __C=0, VV=00, T=0000) interact with the same generic ket?
#    Note: This is Noun * Noun composition using the _general_compose rule.
operator_n_cc0_vv0_t0 = create_noun_from_morphology(cc=0b0, vv=0b00, t=0b0000)
explore_operator_ket_t_effect(operator_n_cc0_vv0_t0, generic_noun_ket_prefix_value)

# 4. Explore a C=0 (Noun) operator with __C=1, VV=11, T=1010
#    How does BW(C:0, __C=1, VV=11, T=1010) interact with the same generic ket?
operator_n_cc1_vv3_t10 = create_noun_from_morphology(cc=0b1, vv=0b11, t=0b1010)
explore_operator_ket_t_effect(operator_n_cc1_vv3_t10, generic_noun_ket_prefix_value)

# 5. Explore a C=1 (Verb) operator with V=010, T=0101
#    How does BW(C:1, V=010, T=0101) interact with a C=1, V=000 ket template (0b1000xxxx)?
operator_v2_t5 = create_verb_from_morphology(v=0b010, t=0b0101)
generic_verb_ket_prefix_value = 0b10000000 # C=1, V=000, T=xxxx
explore_operator_ket_t_effect(operator_v2_t5, generic_verb_ket_prefix_value)

# You can add more examples here to explore different operator morphologies
# and different ket template prefixes.