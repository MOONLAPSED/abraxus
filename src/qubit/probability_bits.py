from __future__ import print_function
import sys
import math
import random
import string


"""
Probabilistic Byte in Information Theory

Consider a byte where each bit has an independent probability p of being 1.
We can calculate the probability of the byte taking any specific value.

Example:
Let p = 0.6 (probability of a bit being 1)
Calculate P(10101010)

P(10101010) = P(1) * P(0) * P(1) * P(0) * P(1) * P(0) * P(1) * P(0)
             = p * (1-p) * p * (1-p) * p * (1-p) * p * (1-p)
             = 0.6 * 0.4 * 0.6 * 0.4 * 0.6 * 0.4 * 0.6 * 0.4

This calculation gives us the probability of observing the specific byte value 10101010
given the probabilistic nature of each bit.
"""

def bit_probability(bit, p):
    return p if bit == 1 else 1 - p

def byte_probability(byte_str, p):
    return math.prod([bit_probability(int(bit), p) for bit in byte_str])


# Example usage
p = 0.6
byte_str = "10101010"
probability = byte_probability(byte_str, p)
print(f"P({byte_str}) = {probability:.8f}")

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
