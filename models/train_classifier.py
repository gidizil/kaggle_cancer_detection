import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.basic_two_layers_model import Net
import configparser


"""==========================================="""
""" Create a class for Training/Testing       """
""" classifier given a training Loader object """
""" based on architectures defined in models/ """
""" use config.txt to define hyperparameters  """
"""==========================================="""


class Classifier:

    """====================================================="""
    """ Class that inherits a model (architecture) instance """
    """ and has two main methods: 'fit' and predict'.       """
    """ Setting learning/testing params is also supported   """
    """ In addition also cares for GPU support              """
    """====================================================="""

    # TODO: handle params using **kwargs
    def __init__(self, classifier):
        self.classifier = classifier
        # TODO: handling **kwargs

        self.epochs = None

        self._set_device()
        self.pass_device_to_gpu()

    def _set_device(self):
        """set device to gpu if possible"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def pass_device_to_gpu(self):
        """if device is gpu - support multiple gpu"""
        if torch.cuda.is_available():
            self.classifier = nn.DataParallel(self.classifier)
            self.epochs = 10
        else:
            self.epochs = 3

        self.classifier = self.classifier.to(self.device)

    #TODO: work with kwargs to enable more params
    def set_loss_function(self):
        """setting loss function and optimization method."""
        # TODO: allow things that aren't adam
        criterion = nn.CrossEntropyLoss()
        optimizier = optim.Adam(self.classifier.parameters(),
                                lr=0.001)
        return criterion, optimizier

    def fit(self, train_loader):
        """ Train classifier. please provide train loader"""
        criterion, optimizier = self.set_loss_function()
        torch.manual_seed(42)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader, 0):
                # set params and all others to train mode
                self.classifier.train()
                # pass data gpu (if exists)
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                scores = self.classifier(images)
                loss = criterion(scores, labels)

                # Backward pass
                optimizier.zero_grad()
                loss.backward()
                optimizier.step()

                running_loss += loss.item()

                # print average loss in last few steps
                if i % 10 == 9:
                    # TODO: do this in .format notation
                    print('Epoch: {0}, Batch: {1}, Loss: {2:.2f}'.format(epoch + 1, i + 1, running_loss))
                    running_loss = 0.0

        print('Training Finished!')

    def predict(self, test_loader):
        """ Test classifier. please provide test loader"""
        correct = 0
        total = 0
        self.classifier.eval() # set to architecture to test mode (BN, dropout etc.)
        with torch.no_grad():
            for (images, labels) in test_loader:
                outputs = self.classifier(images)
                _, predicted = torch.max(outputs, 1) # get highest score classification
                total += labels.size(0)
                correct += (predicted == labels).sum().item() # .item() neccassery


    # TODO: add precision recall support








