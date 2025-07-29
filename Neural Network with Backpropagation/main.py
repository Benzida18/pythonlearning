import numpy as np

# XOR inputs and outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(1)

input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1

weights_input_hidden = 2 * np.random.random((input_layer_neurons, hidden_layer_neurons)) - 1
weights_hidden_output = 2 * np.random.random((hidden_layer_neurons, output_neurons)) - 1

for epoch in range(10000):
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output)
    final_output = sigmoid(final_input)

    output_error = y - final_output
    output_delta = output_error * sigmoid_derivative(final_output)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    weights_hidden_output += hidden_output.T.dot(output_delta)
    weights_input_hidden += X.T.dot(hidden_delta)

print("Final outputs after training:")
print(final_output)

print("\nFinal hidden weights:")
print(weights_input_hidden)

print("\nFinal output weights:")
print(weights_hidden_output)

print("\nExpected vs Predicted:")
for i in range(len(X)):
    print(f"Input: {X[i]} → Expected: {y[i][0]}, Predicted: {round(final_output[i][0], 2)}")