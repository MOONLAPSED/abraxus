def encode_probabilities(alpha, beta):
    # Normalize probabilities
    total = alpha + beta
    norm_alpha = int((alpha / total) * 15)
    norm_beta = int((beta / total) * 15)
    
    # Convert to binary and combine into a byte
    alpha_bits = format(norm_alpha, '04b')
    beta_bits = format(norm_beta, '04b')
    return alpha_bits + beta_bits

def decode_probabilities(byte):
    # Split the byte into alpha and beta parts
    alpha_bits = byte[:4]
    beta_bits = byte[4:]
    
    # Convert from binary to decimal
    norm_alpha = int(alpha_bits, 2)
    norm_beta = int(beta_bits, 2)
    
    # Normalize to probabilities
    total = norm_alpha + norm_beta
    alpha = norm_alpha / 15
    beta = norm_beta / 15
    return alpha, beta

# Example usage
encoded_byte = encode_probabilities(0.33, 0.8)
alpha, beta = decode_probabilities(encoded_byte)
print(f"Encoded Byte: {encoded_byte}")
print(f"Decoded Probabilities: alpha = {alpha}, beta = {beta}")
