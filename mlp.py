from util import *
import numpy as np
from numpy.typing import NDArray
import json

class MultilayerPerceptron:

    '''
        Makes use of the idea that each layer is itself a vector, to thus vectorize propagation of input.
        '''
    def __init__(self):

        # Hyperparameters for model
        self.learning_rate = 0.003
        self.iterations = 1

        # Parameters for neural network layers
        self.activation_functions = list() # activation for each hidden layer and the output layer
        self.activation_derivatives = list()
        self.layer_sizes = None
        self.layers = None

        # Plot data
        self.plot_data = list()

    def initialize_mlp(self, num_features: int, num_actions: int):
        '''
            Initializes the layers of the multilayer perceptron.
            @ num_features : number of features/attributes in the input data'''
        
        if self.layer_sizes is None:
            self.layer_sizes = self.layer_sizes = [num_features, 64, 64, num_actions] # sizes for each layer from the input (index 0) to output layer (index n - 1)

            # init activation functions and derivatives to be used for each layer
            for i in range(len(self.layer_sizes) - 1):
                # hidden layers
                self.activation_functions.append(NeuronLayer.tanh) 
                self.activation_derivatives.append(NeuronLayer.tanh_derivative)

            # output layer
            self.activation_functions.append(NeuronLayer.linear) 
            self.activation_derivatives.append(NeuronLayer.tanh_derivative)

        # Creates the layers of the neural network
        self.layers = list()
        for i in range(len(self.layer_sizes) - 2):
            self.layers.append(NeuronLayer(self.layer_sizes[i] + 1, self.layer_sizes[i+1], 
                                           self.activation_functions[i], self.activation_derivatives[i]))
            
        # Add output layer to layers list
        self.layers.append(OutputLayer(self.layer_sizes[-2] + 1, self.layer_sizes[-1], NeuronLayer.linear, self.activation_derivatives[1]))

    def fit(self, x: NDArray, y: NDArray, num_actions: int):
        '''
            Trains the model on the given input dataset.
            @ x : numpy array of input data
            @ y : numpy array of one-hot encoding of the true outputs
            '''

        # normalize the input data
        x = MultilayerPerceptron.normailze_data(x)

        # Grab dimensions of input data
        num_samples, num_features = x.shape # rows and columns of the input data x, respectively
        
        # initialize model layers and weights
        if self.layers == None:
            self.initialize_mlp(num_features, num_actions)
            
        # Propagate the current batch and perform backpropagation
        y_pred = self.predict_normalized(x)
        self.backprop(y_pred, y)
                
    
    def predict_normalized(self, inputs: NDArray):
        '''
            Performs forward propagation with the given input data.
            @ inputs : normalized input data array
            return : prediction of model from forward pass of input
            '''
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
            #print(output)
        return output
    
    def predict(self, input: NDArray):
        '''
            Performs forward propagation with the given input data.
            @ inputs : unnormalized input data array
            return : prediction of model from forward pass of input
            '''
        input = MultilayerPerceptron.normailze_data(input)

        output = input
        for layer in self.layers:
            output = layer.forward(output)
            #print(output)
        return output
    
    def backprop(self, predictions: NDArray, true_labels: NDArray):
        '''
            Performs backpropagation on the current network model.
            @ output_error : numpy array of delta errors
            '''
        
        # compute output error
        error = self.layers[-1].backward(predictions, true_labels, self.learning_rate)

        # Compute delta error for each hidden layer and update weights
        for layer in reversed(self.layers[:-1]):
            error = layer.backward(error, self.learning_rate)

    def save(self, path):
        '''
            Writes this mlp model to a text file.
            '''
        model_data = {
            "learning rate": self.learning_rate,
            "iterations": self.iterations,
            "batch size": self.batch_size,
            "decay rate": self.decay_rate,
            "layer sizes": self.layer_sizes,
            "layer weights": [layer.get_weights().tolist() for layer in self.layers] # layer weights from 0 to output
        }

        with open(path, 'w') as file:
            json.dump(model_data, file)

    def load(path):
        '''
            Loads the given mlp model.
            '''
        with open(path, 'r') as file:
            model_data = json.load(file)
        
        # create new mlp
        loaded_mlp = MultilayerPerceptron()

        # set model parameters to match loaded model data
        loaded_mlp.learning_rate = model_data['learning rate']
        loaded_mlp.iterations = model_data['iterations']
        loaded_mlp.batch_size = model_data['batch size']
        loaded_mlp.decay_rate = model_data['decay rate']
        loaded_mlp.layer_sizes = model_data['layer sizes']

        # init activation functions and derivatives to be used for each layer
        for i in range(len(loaded_mlp.layer_sizes) - 1):
            # hidden layers
            loaded_mlp.activation_functions.append(NeuronLayer.tanh) 
            loaded_mlp.activation_derivatives.append(NeuronLayer.tanh_derivative)

        # output layer
        loaded_mlp.activation_functions.append(NeuronLayer.linear) 
        loaded_mlp.activation_derivatives.append(NeuronLayer.tanh_derivative)

        # create new layers of mlp
        loaded_mlp.layers = list()
        for i in range(len(model_data['layer weights']) - 1):
            loaded_mlp.layers.append(NeuronLayer(loaded_mlp.layer_sizes[i] + 1, loaded_mlp.layer_sizes[i+1], 
                                           loaded_mlp.activation_functions[i], loaded_mlp.activation_derivatives[i]))
            loaded_mlp.layers[-1].set_weights(np.array(model_data['layer weights'][i]))
            
        # Add output layer to layers list
        loaded_mlp.layers.append(OutputLayer(loaded_mlp.layer_sizes[-2] + 1, loaded_mlp.layer_sizes[-1], NeuronLayer.linear, loaded_mlp.activation_derivatives[1]))
        loaded_mlp.layers[-1].set_weights(np.array(model_data['layer weights'][-1]))

        return loaded_mlp
    
    def get_layers(self):
        return self.layers
    
    def set_layers(self, layers):
        self.layers = layers
        
    def print_weights(self):
        '''
        Outputs the model's weights to console.
        '''
        for layer in self.layers:
            print(layer.get_weights())

    @staticmethod
    def normailze_data(data: NDArray):
        """
        Perform min-max normalization on a dataset.
        Each feature (column) is scaled to the range [0, 1] using the formula:
        (x - min) / (max - min)

        @ data: numpy array of data to be normalized
        return : normalized data
        """

        # min_values = data.min(axis=0)
        # max_values = data.max(axis=0)

        # # Avoid division by zero in case max and min values are the same
        # range_values = max_values - min_values
        # range_values[range_values == 0] = 1

        # # Apply the min-max normalization
        # normalized_data = (data - min_values) / range_values
        # return normalized_data
        return data
    
class NeuronLayer:

    def __init__(self, num_incoming_connections, num_neurons, activation_func, deriv_activation=None):
        self.weights = np.random.randn(num_neurons, num_incoming_connections) * (1/np.sqrt(num_incoming_connections)) # num units in layer X num weights to layer, +1 for bias
        self.activation_func = activation_func
        self.deriv_func = deriv_activation

    def forward(self, input: NDArray):
        '''
            Performs feed-forward propagation with the given input
            @input : numpy array of input data
            return : output from the model based on the given input
            '''
        # Add a column of ones to the input data for the bias
        bias_input = np.hstack([input, np.ones((input.shape[0], 1))])

        self.input = bias_input
        self.output = self.activation_func(np.dot(self.input, self.weights.T))
        return self.output
    
    def backward(self, output_error: NDArray, learning_rate: float):
        '''
            Performs delta error backpropagation for the current layer.
            @ output_error : errors from the following layer
            @ learning_rate: learning rate of the model
            return : array of errors for previous layer input
            '''

        # compute the error and gradient for the current layer 
        error = self.deriv_func(self.output) * output_error
        gradient = np.dot(error.T, self.input)

        # calculate input error for previous layer
        input_error = np.dot(error, self.weights)[:, :-1]

        # update the weights for this layer
        self.weights -= learning_rate * gradient

        return input_error
    
    @staticmethod
    def tanh(z: NDArray):
        '''
            Computes the tanh function on a numpy array z.
            @ z : numpy array from the dot product of weights and input data
            @ return : output from tanh function
            '''
        #return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z: NDArray):
        '''
            Compute the tanh derivative on the given input array z.
            @ z : numpy array
            return : output from tanh prime
            '''
        
        return (1 - (np.tanh(z) ** 2))
    
    @staticmethod
    def linear(z: NDArray):
        return z
    
    def print_weights(self):
        print(self.weights.shape)
        print(self.weights)

    def get_weights(self):
        return self.weights
    
    def set_weights(self, weights):
        self.weights = weights

    def layer_unit_type(self):
        return str(self.activation_func)
    
class OutputLayer(NeuronLayer):

    def backward(self, predictions: NDArray, true_labels: NDArray, learning_rate: float):
        '''
            Performs delta error backpropagation for the current layer.
            @ output_error : errors from the following layer
            @ learning_rate: learning rate of the model
            return : array of errors for previous layer input
            '''

        # compute the error and gradient for the current layer 
        output_error = true_labels - predictions
        gradient = np.dot(output_error.T, self.input)

        # Compute regularization term
        #reg_term = lambda_reg * self.weights
        #reg_term[:, -1] = 0 # exclude bias from regularization

        # input error
        input_error = np.dot(output_error, self.weights)[:, :-1]

        # update the weights for this layer
        self.weights -= learning_rate * gradient

        # calculate input error for previous layer
        return input_error