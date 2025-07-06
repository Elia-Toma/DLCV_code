import numpy as np

timesteps = 100 # Number of timesteps in the input sequence
input_features = 32 # Dimensionality of the input feature space
output_features = 64 # Dimensionality of the output feature space

inputs = np.random.random((timesteps, input_features)) # Input data: random noise for the sake of the example

state_t = np.zeros((output_features,)) # Initial state: an all-zero vector

# Creates random weight matrices
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []

for input_t in inputs: # input_t is a vector of shape (input_features,).
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) # Combines the input with the current state (the previous output) to obtain the current output

    successive_outputs.append(output_t) # Stores this output in a list

    state_t = output_t # Updates the state of the network for the next timestep

final_output_sequence = np.concatenate(successive_outputs, axis=0) # The final output is a 2D tensor of shape (timesteps, output_features).