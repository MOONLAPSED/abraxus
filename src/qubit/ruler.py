import math

class ScaleNormalizer:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale  # e.g., Planck length
        self.max_scale = max_scale  # e.g., Universe diameter
        self.log_range = math.log(max_scale / min_scale)

    def normalize(self, value):
        if value <= 0:
            raise ValueError("Value must be positive")
        log_value = math.log(value / self.min_scale)
        normalized = log_value / self.log_range
        return min(255, max(0, int(normalized * 255)))

    def confidence(self, value, uncertainty):
        relative_uncertainty = uncertainty / value
        return min(15, max(0, int(15 * (1 - relative_uncertainty))))

    def to_color(self, byte_value):
        # Simple RGB mapping (you could use a more sophisticated color map)
        r = 255 - byte_value
        b = byte_value
        g = 255 - abs(128 - byte_value)
        return f"#{r:02x}{g:02x}{b:02x}"

    def represent(self, value, uncertainty):
        byte_value = self.normalize(value)
        conf = self.confidence(value, uncertainty)
        color = self.to_color(byte_value)
        return byte_value, conf, color

# Usage
normalizer = ScaleNormalizer(1e-35, 8.8e26)  # Planck length to observable universe

# Example: Represent the size of an atom (~ 1e-10 m)
atom_size, atom_conf, atom_color = normalizer.represent(1e-10, 1e-12)
print(f"Atom: Byte={atom_size}, Confidence={atom_conf}, Color={atom_color}")

# Example: Represent the size of the Earth (~ 1.2e7 m)
earth_size, earth_conf, earth_color = normalizer.represent(1.2e7, 1e3)
print(f"Earth: Byte={earth_size}, Confidence={earth_conf}, Color={earth_color}")