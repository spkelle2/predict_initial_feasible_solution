import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class FCLayer(Layer):
    """ A fully connected neural network layer, i.e. every possible edge from
    one activation layer to the next"""
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters - stochastic gradient with fixed step size
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer(Layer):
    """ An activation layer in a neural network, i.e. the column of nodes between
    two sets of edges"""
    def __init__(self, activation, activation_prime):
        self.activation = activation  # the activation function to use
        self.activation_prime = activation_prime  # activation function derivative

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


class ActivationFunction:

    @staticmethod
    def f(x):
        return np.tanh(x)

    @staticmethod
    def g(x):
        return 1-np.tanh(x)**2


class LossFunction:
    """Evaluation of Maximum Likelihood Estimation (11) from 'Solving Mixed
    Integer Programs Using Neural Networks' """

    @staticmethod
    def f(mip_sol, nn_out):
        return np.sum(mip_sol * np.log(np.exp(-nn_out) + 1) + (1 - mip_sol) * np.log(np.exp(nn_out) + 1))

    @staticmethod
    def g(mip_sol, nn_out):
        return mip_sol + 1/(np.exp(nn_out) + 1) - 1


class Network:
    """ Neural Network class tying all the above classes together """
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss  # the loss function to use
        self.loss_prime = loss_prime  # its derivative

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, train, var_val_cols, epochs, learning_rate):
        # sample dimension first
        samples = len(train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(train[j][var_val_cols], output)

                # backward propagation
                error = self.loss_prime(train[j][var_val_cols], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))