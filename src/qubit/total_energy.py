import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

def calculate_energy(embedding, context_embedding):
    # Example energy calculation: norm of the embedding vector adjusted by context
    energy = torch.norm(embedding).item()
    context_energy = torch.norm(context_embedding).item()
    total_energy = energy + context_energy
    return total_energy

def get_token_energies(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt')
    
    # Get the hidden states from the model
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    
    # Extract the embeddings for each token from the last hidden state
    last_hidden_state = hidden_states[-1][0]
    
    # Calculate the energy for each token considering its context
    token_energies = []
    for i, token_embedding in enumerate(last_hidden_state):
        if i == 0 or i == len(last_hidden_state) - 1:
            context_embedding = token_embedding
        else:
            context_embedding = last_hidden_state[i-1] + last_hidden_state[i+1]
        energy = calculate_energy(token_embedding, context_embedding)
        token_energies.append(energy)
    
    # Decode tokens to get readable text
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Combine tokens with their energy values
    token_energy_pairs = list(zip(tokens, token_energies))
    
    return token_energy_pairs

# Example usage
text = "Proteins tend to minimize potential energy."
token_energy_pairs = get_token_energies(text)

for token, energy in token_energy_pairs:
    print(f"Token: {token}, Energy: {energy}")
