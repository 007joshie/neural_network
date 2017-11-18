import sys
import numpy as np
import math

class dataset_handler:
    """
    Handles a few different datasets and puts them in readable format by the
    neural_net class.
    """
    def __init__(self, dataset):
        if dataset == 'iris':
            # loads the datasets directly
            from sklearn.datasets import load_iris
            iris = load_iris()

            self.data = iris.data

            # puts the targets in node-like format
            self.output_labels = np.zeros((len(self.data), 3)) #3 is the number of flowers
            for i in range(len(self.data)):
                self.output_labels[i][iris.target[i]] = 1

            print ("dataset '{}' loaded".format(dataset)) #debugging method
        elif dataset == 'mnist':

            #loads the datasets directly
            from mnist import MNIST
            mndata = MNIST('/Users/aidwen/mnist_files')
            self.data, mnist_targets = mndata.load_training()

            #puts the targets in node-like format
            self.output_labels = np.zeros((len(self.data), 10)) #10 is the number of digits
            for i in range(len(self.data)):
                self.output_labels[i][mnist_targets[i]] = 1
            #debugging method
            print ("dataset '{}' loaded".format(dataset))
        else:
            raise Exception("Unknown dataset '{}'.".format(dataset))

class neural_net:
    def sigmoid(self, inputs, in_layer, out_layer):
        """Function used to implement a sigmoid node, which uses the sigmoid
        (mathematical) function"""
        # get the weights and biases of only one node
        sigmoid_weights = self.weights[in_layer][out_layer]
        sigmoid_bias = self.bias[in_layer][out_layer]

        z = 0
        for i in range(0, len(inputs)):
            # inputs in this case are the outputs from the previous nodes
            z += inputs[i] * sigmoid_weights[i]
        z += sigmoid_bias

        output = 1/(1+ math.exp(-z)) #the sigmoid function
        return output

    def sigmoid_prime(self, z):
        """The derivitive of the sigmoid function, which can be used to
        calclate the origional result from the output"""
        # Note that z here is the output from the sigmoid function.
        return (z) * (1-(z))

    def __init__(self, layers, step_size):
        """Inits class with weights and biases.

        Weights are stored as a list of 2d numpy arrays
            [# of layers minus 1][layer i + 1][layer i]
        Biases are stored as a list of 1d numpy arrays
            [# of layers][layer i]
        Args:
            layers: list of the intended network layers.
            step_size: learning rate
            rand_func: function pointer to a randomization function that will
                set the weights and bias default values.

        Returns:
            None
        """
        #just some variables that are used in a few functions. Self-explanatory.
        self.learn_iteration = 0 #how many generations have passed?
        self.layers = layers # list of layers, ie. [4, 5, 3]
        self.network_depth = len(layers) # saves computation time

        self.step_size = step_size # how much to adjust by/the speed of the network

        #initializes the weights array
        self.weights = [np.empty([self.layers[i + 1], self.layers[i]]) for i in range(0, self.network_depth-1)]
        #fills it with normally distributed values, mean of 0, stdev of 0.1
        for i in range(0, self.network_depth - 1):
            for j in range(0, self.layers[i + 1]):
                for k in range(0, self.layers[i]):
                    self.weights[i][j][k] = np.random.normal(0, 0.1)

        #simutaniously creates and fills the bias array with normally distributed values.
        self.bias = []
        for layer_index in range(1, self.network_depth):
            bias_layer = np.empty(self.layers[layer_index])
            for i in range(0, self.layers[layer_index]):
                bias_layer[i] = np.random.normal(0, 0.1)
            self.bias.append(bias_layer)

    def forward_pass(self, input_list):
        """Passes the inputs through the network of weights and biases

        Args:
            input_list: list of the inputs. inputs are in the form of lists with
            length equal to the number of input nodes, or the number of nodes in
            the first layer

        Returns:
            output_list: list of outputs gathered from passing the inputs
            through the network. outputs are in the form of lists with length
            equal to the number of output nodes, or the number of nodes in the
            last layer.
        """
        self.num_inputs = len(input_list) # saves computation time

        #defines an empty output list
        output_list = []
        for i in range(self.num_inputs):
            output = []
            for node in self.layers:
                output_layer = np.empty(node)
                output.append(output_layer)
            output_list.append(output)

        for i in range(0, self.num_inputs):
            for j in range(1, self.network_depth):
                if j == 1: #if there are no nodes before this layer
                    inputs = input_list[i] #use the input data
                    output_list[i][0] = input_list[i]
                else:
                    inputs = output_list[i][j - 1] # otherwise use the output from the previous nodes
                for k in range(0, self.layers[j]):
                    output_list[i][j][k] = self.sigmoid(inputs, j - 1, k) #see sigmoid function above
        return output_list

    def get_error(self, output_list, output_labels):
        """Finds the error for an individual element of the input list

        Called by the backpropogate function.
        """
        #define an empty error list to fill with the errors
        error_list = [np.empty(self.layers[-j - 1]) for j in range(self.network_depth - 1)]
        #calculates the difference between the expected output and the acctual output
        output_difference = output_list[-1] - output_labels
        #multiplies each of the elements of the matricies together (not matrix multiplication)
        error_list[-1] = output_difference * self.sigmoid_prime(output_list[-1])

        #calculates the weights' effect on the last layer, then reverses it
        weights_factor = 0
        for k in range(0, 3):
            weights_factor +=(error_list[-1][k] * self.weights[1][k][0])

        #multiplies each of the elements of the matricies together (not matrix multiplication)
        error_list[-2] = weights_factor * self.sigmoid_prime(output_list[-2])
        return error_list

    def backpropogate(self, output_list, output_labels):
        """
        Manages the adjusting of weights and biases. Calculates error with get_error.
        """
        #defining empty arrays and lists
        error_array = []
        bias_adjust = [np.zeros(self.layers[i+1]) for i in range(self.network_depth - 1)]
        weights_adjust = [np.zeros([self.layers[i], self.layers[i + 1]]) for i in range(0, self.network_depth - 1)]

        for i in range(self.num_inputs):
            #calls get_error(), appends to list
            error_array.append(self.get_error(output_list[i], output_labels[i]))

            #only needs to backpropogate twice; the first layer can have no error
            for j in range(self.network_depth - 1):

                bias_adjust[j] += error_array[i][j]

                #formatting the matricies
                output_matrix = np.transpose(output_list[i][j].reshape(1, len(output_list[i][j])))
                error_matrix = error_array[i][j].reshape(1, len(error_array[i][j]))

                #np.dot() is the mathematical matrix multiplication
                weights_adjust[j] += np.dot(output_matrix, error_matrix)

        for i in range(self.network_depth - 1):
            #adjust the adjustions according to speed
            bias_adjust[i] = self.step_size * (bias_adjust[i] / self.num_inputs)
            weights_adjust[i] = self.step_size * (weights_adjust[i] / self.num_inputs)
            #adjust
            self.bias[i] -= bias_adjust[i]
            self.weights[i] -= np.transpose(weights_adjust[i])


    def evaluate(self, output_list, output_labels, round_num):
        """
        Evaluates how many items the network got correct.
        The network give an output in terms of likelyhood, so this function uses
        the highest likelyhood to determine the network's choice.
        """
        #creates an array of zeros, one row per input, and rows to fit the potential answers
        activation = np.zeros((self.num_inputs, self.layers[-1]))
        #for every element in num_inputs
        for i in range(self.num_inputs):
            max_num = output_list[i][-1][0] #finds the maximum number the network outputted
            saved_index = 0
            for j in range(1, self.layers[-1]):
                if output_list[i][-1][j] > max_num:
                    saved_index = j
                    max_num = output_list[i][-1][j]
            activation[i][saved_index] = 1

        num_correct = 0
        #saves the corrects and incorrects as pairs of length self.layers[-1] arrays
        self.corrects = []
        self.incorrects = []

        for i in range(0, self.num_inputs): #counts total number correct and appends to the lists
            correct = False
            for j in range(0, self.layers[-1]):
                if activation[i][j] == output_labels[i][j] == 1:
                    correct = True
            if correct == True:
                num_correct += 1
                self.corrects.append([activation[i][-1], output_labels[i]])
            else:
                self.incorrects.append([activation[i][-1], output_labels[i]])
        #print output
        print("Iteration %i: %i / %i correct" % (round_num, num_correct, self.num_inputs))
        return activation, num_correct

    def learn(self, data, output_labels, generation_size):
        """
        Two-part function.
        First runs each of the above functions in a working order, and manages
        how many iterations before a generation ends and the second part runs.
        """
        for iteration in range(0, generation_size):
            output_list = self.forward_pass(data)
            learn_iteration = self.learn_iteration * generation_size + iteration + 1
            activation, num_correct = self.evaluate(output_list, output_labels, learn_iteration)
            self.backpropogate(output_list, output_labels)
        self.learn_iteration += 1

        """
        Second part generates data on the network output. Useful for debugging.
        Also you can see how close some incorrect and correct pairings are.
        """

        for i in range(0, self.num_inputs):
            grade = 'False'
            if np.array_equal(output_labels[i], activation[i]):
                grade = 'True '
            self.print_table(grade, data[i], output_labels[i], activation[i], output_list[i][-1])
        self.print_summary(num_correct, generation_size)
        confirm = raw_input("Continue (RETURN) or Cancel(ANY)? : ")
        if confirm == '':
            self.learn(data, output_labels, generation_size)


    def print_table(self, grade, data, output_labels, activation, output_list):
        print ("Correct? {}   Data: {}   Expected Output: {}   Network Output: {}   Raw Output: {}".format(grade, data, output_labels, activation, output_list))

    def print_summary(self, num_correct, generation_size):
        print ("Generation #{}: {}/{} correct   Generation Size: {}".format(self.learn_iteration, num_correct, self.num_inputs, generation_size))
class main:
    input_dataset = dataset_handler('iris')
    network = neural_net([len(input_dataset.data[0]), 15, len(input_dataset.output_labels[0])], 0.5)
    network.learn(input_dataset.data, input_dataset.output_labels, 100)
