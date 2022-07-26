import argparse
import functools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def softmax(x):
    """
    Compute softmax function for a batch of input values.
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """

    softmax_matrix = np.zeros((x.shape[0], x.shape[1]))

    for row_index in range(x.shape[0]):

        # Use m as normalizing constant s.t. exp will not be too big
        m = np.max(x.iloc[row_index, :])
        denom = np.sum(np.exp(x.iloc[row_index, :] - m))

        for col_index in range(x.shape[1]):

            numerator = np.exp(x.iloc[row_index, col_index] - m)
            softmax_matrix[row_index, col_index] = numerator / denom

    return softmax_matrix
    # return np.exp(x - np.max(x, axis=1)) / np.sum(np.exp(x - np.max(x, axis=1)), axis=0)


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x * (x > 0)  # max(0, x)


def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.

    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes

    Returns:
        A dict mapping parameter names to numpy arrays
    """

    W1 = np.random.randn(input_size, num_hidden)
    b1 = np.array([0.] * num_hidden)

    W2 = np.random.randn(num_hidden, num_output)
    b2 = np.array([0.] * num_output)

    return {'W2': W2, 'W1': W1, 'b1': b1, 'b2': b2}


def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """

    z_1 = data @ params['W1'] + params['b1']
    a_1 = sigmoid(z_1)

    z_2 = a_1 @ params['W2'] + params['b2']
    output = softmax(z_2)

    CE = 0
    for i in range(len(labels)):
        CE -= labels[i].dot(np.log(output[i])) / len(labels)

    return a_1, output, CE


def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """

    a_1, output, cost = forward_prop_func(data, labels, params)

    d_Z2 = output - labels
    d_W2 = a_1.T @ d_Z2
    d_b2 = np.sum(d_Z2, axis=0, keepdims=True)

    d_Z1 = np.array(d_Z2 @ params['W2'].T) * (a_1 * (1 - a_1))
    d_W1 = data.T @ d_Z1
    d_b1 = np.array(np.sum(d_Z1, axis=0))  # .reshape(1, NUM_HIDDEN)

    return {'W2': d_W2, 'W1': d_W1, 'b1': d_b1.reshape(NUM_HIDDEN, ), 'b2': d_b2.reshape(NUM_OUTPUT, )}


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):

    a_1, output, cost = forward_prop_func(data, labels, params)

    d_Z2 = output - labels
    d_W2 = a_1.T @ d_Z2 + 2 * reg * params['W2']
    d_b2 = np.sum(d_Z2, axis=0, keepdims=True)

    d_Z1 = np.array(d_Z2 @ params['W2'].T) * (a_1 * (1 - a_1))
    d_W1 = (data.T @ d_Z1) + 2 * reg * params['W1']
    d_b1 = np.array(np.sum(d_Z1, axis=0))

    return {'W2': d_W2, 'W1': d_W1, 'b1': d_b1.reshape(NUM_HIDDEN, ), 'b2': d_b2.reshape(NUM_OUTPUT, )}


def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    mini_batches = len(train_data) // batch_size

    for mini_batch_count in range(mini_batches):

        train_data_batch = train_data[batch_size * mini_batch_count: batch_size * (mini_batch_count + 1)]
        train_labels_batch = train_labels[batch_size * mini_batch_count: batch_size * (mini_batch_count + 1)]

        back_prop_batch = backward_prop_func(train_data_batch, train_labels_batch, params, forward_prop_func)

        for param in params:
            params[param] -= learning_rate * (1 / batch_size) * back_prop_batch[param]

    # This function does not return anything
    return


def nn_train(
        train_data, train_labels, get_initial_params_func, forward_prop_func, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):
    """
    Train model using gradient descent for specified number of epochs.

    Evaluates cost and accuracy on training and dev set at the end of each epoch.

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        dev_data: A numpy array containing the dev data
        dev_labels: A numpy array containing the dev labels
        get_initial_params_func: A function to initialize model parameters
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API
        num_hidden: Number of hidden layers
        learning_rate: The learning rate
        num_epochs: Number of epochs to train for
        batch_size: The amount of items to process in each batch

    Returns:
        params: A dict of parameter names to parameter values for the trained model
        cost_train: An array of training costs at the end of each training epoch
        cost_dev: An array of dev set costs at the end of each training epoch
        accuracy_train: An array of training accuracies at the end of each training epoch
        accuracy_dev: An array of dev set accuracies at the end of each training epoch
    """

    (nexp, dim) = train_data.shape

    num_output = NUM_OUTPUT
    params = get_initial_params_func(dim, num_hidden, num_output)

    cost_train = []
    accuracy_train = []

    LOW_CHANGE = 0

    for epoch in range(num_epochs):

        # Falling Learning rate later to help convergence
        if DECREASE_lR and epoch > 30:
            if learning_rate > 3:
                learning_rate = learning_rate * 0.99

        gradient_descent_epoch(train_data, train_labels,
                               learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output, train_labels))

        print(epoch, "- Loss:", cost, "- Accuracy:", accuracy_train[-1])
        if epoch != 0:
            acc_change = accuracy_train[-1] / accuracy_train[-2] - 1
            if abs(acc_change) < CONV_THRESHOLD:

                if accuracy_train[-1] == accuracy_train[-2]:
                    print("SAME")
                    continue

                print(f"LOW CHANGE ({LOW_CHANGE + 1}): Acc change was: {acc_change}")
                LOW_CHANGE += 1
                if LOW_CHANGE >= 3:
                    break

    return params, cost_train, accuracy_train


def nn_test(data, labels, params):
    """Predict labels and compute accuracy for held-out test data"""
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) == np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy


def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    """Trains model, applies model to test data, and (optionally) plots loss"""
    params, cost_train, accuracy_train = nn_train(
        all_data['train'], all_labels['train'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=NUM_HIDDEN, learning_rate=LEARNING_RATE, num_epochs=num_epochs, batch_size=BATCH_SIZE
    )

    t = np.arange(num_epochs)

    cost_train = cost_train + [cost_train[-1]]*(num_epochs - len(cost_train))
    accuracy_train = accuracy_train + [accuracy_train[-1]] * (num_epochs - len(accuracy_train))

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train, 'r', label='train')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train, 'r', label='train')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig(f"./{name}_{NUM_EPOCHS}_{NUM_HIDDEN}_{LEARNING_RATE}.pdf")

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f on TEST data' % (name, accuracy))

    # Export fitted params
    export_params(params, name)

    return accuracy


def export_params(params, name):
    print("INFO: Parameters exported")
    with open(f'params_{name}.pkl', 'wb') as f:
        pickle.dump(params, f)


def load_params(name):
    with open(f'params_{name}.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    print("INFO: Parameters loaded")
    return loaded_dict


def one_hot_labels(labels):
    """Convert labels from integers to one hot encoding"""
    one_hot_labels = np.zeros((labels.size, NUM_OUTPUT))
    # one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1

    for r, label in enumerate(labels.values):
        one_hot_labels[r, label[0]] = 1

    return one_hot_labels


def one_hot_encode(data):
    """ One hot encode a vector """
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = label_encoder.fit_transform(data)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded


def read_data(data_file, labels_file):
    """ Load data and labels """
    x = pd.read_csv(data_file, delimiter=',')
    y = pd.read_csv(labels_file, delimiter=',')
    return x, y


def split_train_test(data_file, ratio=0.8):

    data_file = pd.DataFrame(data_file)

    length = round(data_file.shape[0] * ratio)
    train = data_file.iloc[:length, :]
    test = data_file.iloc[length:, :]
    return train, test


def main(plot=True):

    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)

    args = parser.parse_args()

    data = pd.read_csv("data//bid_censored_simulated.csv")

    labels = data.iloc[:, -1]
    data = data.iloc[:, :-1]

    data = data.loc[:, ["alpha", "beta", "v", "omega"]]

    train_data, test_data = split_train_test(data, 0.8)
    train_labels, test_labels = split_train_test(labels, 0.8)

    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    # Z Feature Scaling (Standardization) - Every feature now follows N(0,1)
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    print(train_data.shape)

    all_data = {
        'train': train_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'test': test_labels
    }

    # baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    baseline_acc = ""
    reg_acc = run_train_test('regularized', all_data, all_labels,
        functools.partial(backward_prop_regularized, reg=REG),
        args.num_epochs, plot)

    return baseline_acc, reg_acc


if __name__ == '__main__':

    NUM_HIDDEN = 3
    NUM_OUTPUT = 2
    LEARNING_RATE = 4
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    REG = 0.0002
    CONV_THRESHOLD = 0.0001

    DECREASE_lR = True

    main(plot=True)
