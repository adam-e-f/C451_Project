# IMPORTS ======================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import keras                        # for datasets
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
    (x_train_digits, y_train_digits), (x_test_digits, y_test_digits)     = keras.datasets.mnist.load_data()
    (x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = keras.datasets.fashion_mnist.load_data()

    return (x_train_digits, y_train_digits), (x_test_digits, y_test_digits), (x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion)

def split_and_create_loader(x,y, batch_size):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.16333333, random_state=5612)
        
        training_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]), dtype=torch.float32), torch.tensor(one_hot(y_train), dtype=torch.float32))
        validation_dataset = torch.utils.data.TensorDataset(torch.tensor(x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2]), dtype=torch.float32), torch.tensor(one_hot(y_val), dtype=torch.float32))

        # data loaders
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, validation_loader

def train_neural_network(neural_network, x_train, y_train):
    neural_network.train(x_train, y_train)

def test_neural_network(neural_network, x_test, y_text):
    return 0
#===============================================================================

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
        x = self.activation_function(self.fully_connected_layer_1(x))
        x = self.activation_function(self.fully_connected_layer_2(x))
        x = self.activation_function(self.fully_connected_layer_3(x))
        return x
    
    def predict (self, x):
        outputs = self(x)
        _, predicted = torch.max(outputs.data, 1)
        return predicted
    
    def validate (self, validation_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validation_loader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                labels_int = one_hot_inverse(labels)
                correct += (predicted == labels_int).sum().item()

        return (100 * correct / total)
    
    def train(self, x, y, max_epochs=100, batch_size=50):
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        number_of_batches = (x.shape[0] // batch_size)
        
        train_loader, validation_loader = split_and_create_loader(x, y, batch_size)

        val_percent = 0
        for epoch in range(max_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            val_percent = self.validate(validation_loader)

            file = open("out.txt", "a")
            file.write(f'epoch: {epoch + 1}, loss: {running_loss / number_of_batches:.4e}, val accuracy: {val_percent:.2f}\n')
            file.close()

            print('[%d, %5d] loss: %.4e, validation accuracy %.3f' % (epoch + 1, i + 1, running_loss / number_of_batches, val_percent))
            running_loss = 0.
        
        #print('Finished Training')
        return running_loss / number_of_batches, val_percent

class Convolutional_Neural_Network(nn.Module):
    def __init__(self, learning_rate, activation_function):
        dense_width = 100

        super(Convolutional_Neural_Network, self).__init__()
        # 2 convolution layers
        self.convolution_layer_1 = nn.Conv2d(1, 6, 5)
        self.pool_layer_1 = nn.MaxPool2d(3, 2)
        self.convolution_layer_2 = nn.Conv2d(6, 16, 3)
        self.pool_layer_2 = nn.MaxPool2d(3, 2)
        x = x.view(-1, 16 * 5 * 5)
        # 2 connected layers
        self.fully_connected_layer_1 = nn.Linear(dense_width, dense_width)
        self.fully_connected_layer_2 = nn.Linear(dense_width, 10)
        
        self.learning_rate = learning_rate
        self.activation_function = activation_function

# Defining main function
def main():
    (x_train_digits, y_train_digits), (x_test_digits, y_test_digits), (x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = load_data()

    deep_net = Deep_Neural_Network(0.1,relu)
    #conv_net = Convolutional_Neural_Network(0.1,relu)

    # train and test on mnist digits data
    file = open("out.txt", "a")
    file.write('Deep Network with MNIST digits data -- training\n')
    file.close()
    train_neural_network(deep_net, x_train_digits, y_train_digits)

    file = open("out.txt", "a")
    file.write('Deep Network with MNIST digits data -- testing\n')
    file.close()
    test_neural_network(deep_net, x_test_digits, y_test_digits)

    # file = open("out.txt", "a")
    # file.write('Conv Network with MNIST digits data -- training\n')
    # file.close()
    # train_neural_network(conv_net, x_train_digits, y_train_digits)

    # file = open("out.txt", "a")
    # file.write('Conv Network with MNIST digits data -- testing\n')
    # file.close()
    # test_neural_network(conv_net, x_test_digits, y_test_digits)

    # train and test on mnist fashion data
    # file = open("out.txt", "a")
    # file.write('Deep Network with MNIST fashion data -- training\n')
    # file.close()
    # train_neural_network(deep_net, x_train_fashion, y_train_fashion)
    # file = open("out.txt", "a")
    # file.write('Deep Network with MNIST fashion data -- testing\n')
    # file.close()
    # test_neural_network(deep_net, x_test_fashion, y_test_fashion)

    # file = open("out.txt", "a")
    # file.write('Conv Network with MNIST fashion data -- training\n')
    # file.close()
    # train_neural_network(conv_net, x_train_fashion, y_train_fashion)

    # file = open("out.txt", "a")
    # file.write('Conv Network with MNIST fashion data -- testing\n')
    # file.close()
    # test_neural_network(conv_net, x_test_fashion, y_test_fashion)

if __name__=="__main__":
    main()