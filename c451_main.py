# IMPORTS ======================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import keras  # for datasets
from sklearn.model_selection import train_test_split


# HELPERS ======================================================================
def sigmoid(x):
    return F.sigmoid(x)

def relu(x):
    return F.relu(x)

def elu(x):
    return F.elu(x)

def tanh(x):
    return F.tanh(x)

def one_hot(vector):
    return np.eye(10)[vector]

def one_hot_inverse(vector):
    v = np.argmax(vector, axis=1)
    return v

def load_data():
    # uses Keras library and Google API to download data
    (x_train_digits, y_train_digits), (x_test_digits, y_test_digits) = keras.datasets.mnist.load_data()
    (x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = keras.datasets.fashion_mnist.load_data()

    return (x_train_digits, y_train_digits), (x_test_digits, y_test_digits), (x_train_fashion, y_train_fashion), (
    x_test_fashion, y_test_fashion)

def split_and_create_loader(x, y, batch_size):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.16333333, random_state=5612)

    training_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train.reshape(-1, 1, 28, 28) / 255.0, dtype=torch.float32),  # Updated for CNN input shape
        torch.tensor(y_train, dtype=torch.long))
    validation_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_val.reshape(-1, 1, 28, 28) / 255.0, dtype=torch.float32),  # Updated for CNN input shape
        torch.tensor(y_val, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader

def train_neural_network(neural_network, x_train, y_train):
    neural_network.train(x_train, y_train)

def test_neural_network(neural_network, x_test, y_test):
    # Check if neural network expects 2D or 4D input.
    if isinstance(neural_network, Convolutional_Neural_Network):
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_test.reshape(-1, 1, 28, 28) / 255.0, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
    else:
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_test.reshape(-1, 28 * 28) / 255.0, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)
    return neural_network.validate(test_loader)


# ===============================================================================

class Deep_Neural_Network(nn.Module):
    def __init__(self, learning_rate, activation_function):
        dense_width = 100

        super(Deep_Neural_Network, self).__init__()

        # 2 connected layers
        self.fully_connected_layer_1 = nn.Linear(28 * 28, dense_width)
        self.fully_connected_layer_2 = nn.Linear(dense_width, dense_width)
        self.fully_connected_layer_3 = nn.Linear(dense_width, 10)

        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.activation_function(self.fully_connected_layer_1(x))
        x = self.activation_function(self.fully_connected_layer_2(x))
        x = self.fully_connected_layer_3(x)
        return x

    def predict(self, x):
        outputs = self(x)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

    def validate(self, validation_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validation_loader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).type(torch.int).sum().item()

        return (100 * correct / total)

    def train(self, x, y, max_epochs=100, batch_size=50):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        number_of_batches = (x.shape[0] // batch_size)

        train_loader, validation_loader = split_and_create_loader(x, y, batch_size)

        val_percent = 0

        with open("out.txt", "a") as file:

            for epoch in range(max_epochs):
                running_loss = 0.0
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data

                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                val_percent = self.validate(validation_loader)

                # file.write(
                #    f'epoch: {epoch + 1}, loss: {running_loss / number_of_batches:.4e}, val accuracy: {val_percent:.2f}\n')

                print('[%d, %5d] loss: %.4e, validation accuracy %.3f' % (
                    epoch + 1, i + 1, running_loss / number_of_batches, val_percent))
                running_loss = 0.

        # print('Finished Training')
        return running_loss / number_of_batches, val_percent


class Convolutional_Neural_Network(nn.Module):
    def __init__(self, learning_rate, activation_function):
        dense_width = 100

        super(Convolutional_Neural_Network, self).__init__()
        # 2 convolution layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        # 2 connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, dense_width)
        self.fc2 = nn.Linear(dense_width, dense_width)
        self.fc3 = nn.Linear(dense_width, 10)

        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def forward(self, x):
        x = self.pool1(self.activation_function(self.conv1(x)))
        x = self.pool2(self.activation_function(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.fc3(x)
        return x

    def validate(self, validation_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validation_loader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).type(torch.int).sum().item()

        return (100 * correct / total)

    def train(self, x, y, max_epochs=100, batch_size=50):
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        train_loader, validation_loader = split_and_create_loader(x, y, batch_size)

        val_percent = 0

        with open("out.txt", "a") as file:

            for epoch in range(max_epochs):
                running_loss = 0.0
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data

                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                val_percent = self.validate(validation_loader)

                # file.write(
                #    f'epoch: {epoch + 1}, loss: {running_loss / len(train_loader):.4e}, val accuracy: {val_percent:.2f}\n')

                print('[%d, %5d] loss: %.4e, validation accuracy %.3f' % (
                    epoch + 1, i + 1, running_loss / len(train_loader), val_percent))

        return running_loss / len(train_loader), val_percent

# Code to train a model on a given dataset and test it. This helper function
# makes it easier to run many trials with variations to the model, dataset, learning
# rate, and activation function.
def run_trail(model, x_train, y_train, x_test, y_test):
    train_neural_network(model, x_train, y_train)
    test_accuracy = test_neural_network(model, x_test, y_test)

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    file = open("out.txt", "a")
    file.write(f"Test Acurracy: {test_accuracy}\n\n")
    file.close()

def main():
    # Open the file in write mode to clear its contents.
    file = open("out.txt", "w")
    file.close()

    learning_rates = [0.01, 0.05, 0.1, 0.2]
    activation_functions = [relu, sigmoid, elu, tanh]

    (x_train_digits, y_train_digits), (x_test_digits, y_test_digits), (x_train_fashion, y_train_fashion), (
    x_test_fashion, y_test_fashion) = load_data()

    # First, run all DNNs on MNIST digits.
    for lr in learning_rates:
        for af in activation_functions:
            file = open("out.txt", "a")
            file.write(f"Deep Network with MNIST digits data -- activation = {af.__name__}, LR = {lr}\n")
            file.close()
            run_trail(Deep_Neural_Network(lr, af), x_train_digits, y_train_digits, x_test_digits, y_test_digits)

    # Next, run all CNNs on MNIST digits.
    for lr in learning_rates:
        for af in activation_functions:
            file = open("out.txt", "a")
            file.write(f"Convolutional Network with MNIST digits data -- activation = {af.__name__}, LR = {lr}\n")
            file.close()
            run_trail(Convolutional_Neural_Network(lr, af), x_train_digits, y_train_digits, x_test_digits, y_test_digits)

    # Next, run all DNNs on MNIST fashion.
    for lr in learning_rates:
        for af in activation_functions:
            file = open("out.txt", "a")
            file.write(f"Deep Network with MNIST fashion data -- activation = {af.__name__}, LR = {lr}\n")
            file.close()
            run_trail(Deep_Neural_Network(lr, af), x_train_fashion, y_train_fashion, x_test_fashion, y_test_fashion)

    # Next, run all CNNs on MNIST fashion.
    for lr in learning_rates:
        for af in activation_functions:
            file = open("out.txt", "a")
            file.write(f"Convolutional Network with MNIST fashion data -- activation = {af.__name__}, LR = {lr}\n")
            file.close()
            run_trail(Convolutional_Neural_Network(lr, af), x_train_fashion, y_train_fashion, x_test_fashion, y_test_fashion)

if __name__ == "__main__":
    main()
