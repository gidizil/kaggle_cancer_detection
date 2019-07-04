import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from general_utils import GPUConfig, HyperParamsConfig
import os


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
        self.class_config = GPUConfig()
        self.path_dict = self.class_config.get_paths_dict()
        self.h_params_config = HyperParamsConfig()
        self.h_params_dict = self.h_params_config.params_dict
        # TODO: handling **kwargs

        self.epochs = None

        self._set_device()
        self.pass_device_to_gpu()

    def _set_device(self):
        """set device to gpu if possible"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def pass_device_to_gpu(self):
        """if device is gpu - support multiple gpu"""
        """ in addition, set fixed seed"""
        if torch.cuda.is_available():
            self.classifier = nn.DataParallel(self.classifier)

        self.epochs = self.h_params_dict['num_epochs']
        self.classifier = self.classifier.to(self.device)

    # TODO: work with kwargs to enable more params
    def set_loss_function(self):
        """setting loss function and optimization method."""
        # TODO: allow things that aren't adam
        if self.h_params_dict['loss_func'] == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif self.h_params_dict['loss_func'] == 'bce':
            criterion = nn.BCELoss()

        optimizier = optim.Adam(self.classifier.parameters())
        return criterion, optimizier

    def fit(self, train_loader):
        """ Train classifier. please provide train loader"""

        criterion, optimizier = self.set_loss_function()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader, 0):
                # set params and all others to train mode
                #TODO: is possible to set classifier outside training loop
                self.classifier.train()
                # pass data gpu (if exists)
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                scores = self.classifier(images)

                if self.h_params_dict['loss'] == 'bce':
                    print(type(labels.view(-1, 1).float()))
                    loss = criterion(scores, labels.view(-1, 1).float())
                elif self.h_params_dict['loss'] == 'cross_entropy':
                    loss = criterion(scores, labels)

                # Backward pass
                optimizier.zero_grad()
                loss.backward()
                optimizier.step()

                running_loss += loss.item()

                # print average loss in last few steps
                if i % 10 == 9:
                    print('Epoch: {0}, Batch: {1}, Loss: {2:.2f}'.format(epoch + 1, i + 1, running_loss))
                    running_loss = 0.0

        print('Training Finished!')

    def predict(self, test_loader):
        """ Test classifier. please provide test loader"""
        correct = 0
        total = 0
        self.classifier.eval()  # set to architecture to test mode (BN, dropout etc.)
        with torch.no_grad():
            for (images, labels) in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.classifier(images)
                _, predicted = torch.max(outputs, 1) # get highest score classification
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    def fit_and_eval(self, train_loader, val_loader):
        """ Method for fit and evaluate loss.   """
        """ Train, and after epoch, evaluate    """
        """ loss on the entire validation set   """
        train_loss_arr = []
        val_loss_arr = []
        accuracy_arr = []
        criterion, optimizier = self.set_loss_function()
        for epoch in range(self.epochs):
            train_loss = 0.0
            val_loss = 0.0
            total_corrcets = 0.0
            total_samples = 0.0
            # train and calculate loss in epoch
            self.classifier.train()
            for i, (images, labels) in enumerate(train_loader):
                # set to train mode
                # TODO: is possible to set classifier outside training loop

                # pass data to gpu:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass:
                optimizier.zero_grad()

                scores = self.classifier(images)
                if self.h_params_dict['loss_func'] == 'bce':
                    labels = labels.view(-1, 1).float()
                    loss = criterion(scores, labels)
                elif self.h_params_dict['loss_func'] == 'cross_entropy':
                    loss = criterion(scores, labels)

                # Backward pass and weight update:
                loss.backward()
                optimizier.step()

                train_loss += loss.item() * labels.size()[0]


            # set classifier to eval mode:
            self.classifier.eval()
            with torch.no_grad():

            #     for name, param in self.classifier.named_parameters():
            #         if param.requires_grad:
            #             print(name, param.data)
            #     for i, (images, labels) in enumerate(val_loader):

                    # pass device to gpu
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    scores = self.classifier(images)
                    if self.h_params_dict['loss_func'] == 'bce':
                        labels = labels.view(-1, 1).float()
                        loss = criterion(scores, labels)
                    elif self.h_params_dict['loss_func'] == 'cross_entropy':
                        loss = criterion(scores, labels)

                    val_loss += loss.item() * labels.size()[0]

                    # Calc Values for accuracy
                    corrects_cnt, samples_cnt = self.calc_accuracy(scores, labels)
                    total_corrcets += corrects_cnt
                    total_samples += samples_cnt

            print('Epoch:', epoch+1, ', Train Loss:', train_loss / len(train_loader.dataset))
            print('Epoch:', epoch+1, ', val Loss:', val_loss / len(val_loader.dataset))
            print('Validation Accuracy:', total_corrcets / total_samples)
            accuracy_arr.append(total_corrcets / total_samples)
            train_loss_arr.append(train_loss / len(train_loader.dataset))
            val_loss_arr.append(val_loss / len(val_loader.dataset))


        # TODO: plot train and val loss for each epoch - Make sure everything goes down
        self.plot_train_val_loss(train_loss_arr, val_loss_arr)
        self.plot_val_accuracy(accuracy_arr)

    def calc_accuracy(self, scores, labels):
        """Calculates the accuracy in the currenct  """
        # classified_indices = scores.argmax(dim=1)
        classified_indices = (scores >= 0.5).float()
        #total_corrects = (classified_indices == labels).sum()
        total_corrects = classified_indices.eq(labels.view_as(classified_indices)).sum().item()
        total_samples = labels.size()[0]

        return total_corrects, total_samples

    # TODO: add precision recall support
    def calc_precision_recall(self, scores, labels):
        """ Calcluate precision and recall for given scores and labels"""
        pass

    def visualize_training(self):
        """Visualize the predictions for images for each label"""
        pass

    def plot_train_val_loss(self, train_loss_arr, val_loss_arr):
        """ Plot the training and validation error after each epoch"""
        plot_path = os.path.join(self.path_dict['plots'], 'train_val_loss.png')
        plt.plot(range(1, self.epochs + 1), train_loss_arr, label='Train Loss')
        plt.plot(range(1, self.epochs + 1), val_loss_arr, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(plot_path)
        plt.close()

    def plot_val_accuracy(self, val_accuracy_arr):
        """Plots the validation accuracy through training"""
        plot_path = os.path.join(self.path_dict['plots'], 'val_accuracy.png')
        plt.plot(range(1, self.epochs + 1), val_accuracy_arr, label='Validation Accuracy')
        plt.legend(loc='upper left')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(plot_path)
        plt.close()










