# Class NFL_NN creates a neural network with a specified number and size of layers that does training and evaluation
# for either regression or classification neural networks.
import torch.nn as nn
import numpy as np


# Parts of this adapted from:
# https://github.com/christianversloot/machine-learning-articles/blob/main
# /how-to-create-a-neural-network-for-regression-with-pytorch.md
class NFL_MLP(nn.Module):
    def __init__(self, data_loaders, num_layers: int, layer_sizes: list[int],
                 activation_function, is_classification: bool):
        super().__init__()
        self.train_loader, self.test_loader = data_loaders
        self.act_f = activation_function
        self.is_classification = is_classification
        # This is a bit rough, can fix up with a loop or ModuleList later
        # self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(num_layers-1)])
        self.layers = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            self.act_f(),
            nn.Linear(layer_sizes[1], layer_sizes[2])
        )
        self.build_layers(num_layers, layer_sizes)

    # Need to adjust a bit, won't work for more layers
    def build_layers(self, num_layers: int, layer_sizes: list[int]):
        if num_layers < 3 or num_layers > 5 or num_layers != len(layer_sizes):
            print(f"ERROR: number of layers: {num_layers}, length of layer_sizes: {len(layer_sizes)}")
        elif num_layers == 4:
            self.layers.append(self.act_f())
            self.layers.append(nn.Linear(layer_sizes[2], layer_sizes[3]))
        elif num_layers == 5:
            self.layers.append(self.act_f())
            self.layers.append(nn.Linear(layer_sizes[3], layer_sizes[4]))
        if self.is_classification:
            self.layers.append(nn.Sigmoid())

    def forward(self, datum):
        return self.layers(datum)

    def train_nn(self, loss_function, optimizer, num_epochs: int, verbose=True):
        numpy_of_outputs = np.zeros((num_epochs, 3))
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()  # don't need this?
                targets = targets.reshape((targets.shape[0], 1))
                optimizer.zero_grad()
                outputs = self(inputs)
                minibatch_loss = loss_function(outputs, targets)
                epoch_loss += minibatch_loss.item()
                minibatch_loss.backward()
                optimizer.step()
            per_game_train_loss = epoch_loss/len(self.train_loader.dataset)
            per_game_test_loss, correct_predictions = self.evaluate(loss_function)
            prediction_accuracy = correct_predictions/len(self.test_loader.dataset)
            if verbose:
                self.print_train_test_loss(epoch, per_game_train_loss, per_game_test_loss, prediction_accuracy)
            numpy_of_outputs[epoch][0], numpy_of_outputs[epoch][1] = per_game_train_loss, per_game_test_loss
            numpy_of_outputs[epoch][2] = prediction_accuracy
        return numpy_of_outputs

    def evaluate(self, loss_function):
        total_loss = 0.0
        correct_predictions = 0.0
        for i, data in enumerate(self.test_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = self(inputs)
            # This comes back two orders of magnitude higher than the above, despite code being nearly identical
            correct_predictions += self.get_correct_predictions(outputs, targets)
            per_batch_test_loss = loss_function(outputs, targets)
            total_loss += per_batch_test_loss.item()
        loss_per_game = total_loss / len(self.test_loader.dataset)
        return loss_per_game, correct_predictions

    def get_correct_predictions(self, outputs, targets):
        num_correct = 0.0
        for output, target in zip(outputs, targets):
            if self.is_classification:
                if output > 0.5 and target == 1.0 or output < 0.5 and target == 0.0:
                    num_correct += 1
            else:
                if output >= 0.0 and target >= 0.0 or output < 0.0 and target < 0.0:
                    num_correct += 1
        return num_correct

    def print_train_test_loss(self, epoch: int, per_game_train_loss: float, per_game_test_loss: float,
                              prediction_accuracy: float):
        print(f"================Training epoch: {epoch}================")
        if self.is_classification:
            print(f"TRAINING: avg loss per game: {per_game_train_loss}")
            print(f"TESTING: avg loss per game: {per_game_test_loss}")
        else:
            print(f"TRAINING: avg points error per game: {per_game_train_loss}")
            print(f"TESTING: avg points loss per game: {per_game_test_loss}")
        print(f"TESTING: games predicted correctly: {prediction_accuracy}")
