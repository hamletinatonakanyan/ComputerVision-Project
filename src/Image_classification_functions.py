# Functions for creating CNN classification model

# import the necessary libraries
import os
import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt


# calculating the amount of training and testing datasets
def get_dataset_length(folder_path):
    """
    :param folder_path: train folder path
    :return: amount of samples of dataset
    """
    dataset_cnt = len(glob.glob(f'{folder_path}/**/*.jpg'))

    return dataset_cnt


# get names and number of output classes
def get_classes(folder_path):
    """
    :param folder_path: train or test folder path
    :return: names and amount of the output classes
    """

    # list for storing names of the output classes
    # variable for keeping number of output classes
    output_classes = []
    num_of_classes = 0

    for path in glob.glob(f'{folder_path}/*'):
        output_classes.append(os.path.split(path)[1])
        num_of_classes = len(output_classes)

    return output_classes, num_of_classes


# check if the images have the same sizes and get sizes and channels
def get_sizes_and_channels(path):
    """
    :param path: takes path of the folder
    :return: different sizes and channels of images
    """
    images_size = [(800, 600)]
    channels = [3]
    for image_path in glob.glob(f'{path}/*/*.jpg'):
        image = Image.open(image_path)
        if len(image.getbands()) != channels[0]:
            channels.append(len(image.getbands()))
        if image.size != images_size[0]:
            images_size.append(image.size)
    return images_size, channels


# Implemented CNN model
class CNN(nn.Module):
    """
    Implement CNN model for image classification
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # input shape (3, 448, 448)
        # output size formula (w-f + 2P)/s + 1

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3)

        # Linear layers
        # after filters and max pooling input shape will be (3, 96, 12, 12)
        self.fc1 = nn.Linear(in_features=96 * 12 * 12, out_features=1024)
        self.out = nn.Linear(in_features=1024, out_features=self.num_classes)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(self.bn1(t), kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.dropout2d(t, p=0.1)

        t = F.relu(self.conv3(t))
        t = F.max_pool2d(self.bn2(t), kernel_size=3, stride=2)

        t = F.relu(self.conv4(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.dropout2d(t, p=0.2)

        t = F.relu(self.conv5(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.fc1(t.reshape(-1, 96 * 12 * 12)))
        t = F.dropout2d(t, p=0.1)
        t = self.out(t)

        return t


# function for freezing layers in transfer learning models
def freeze_until(model, layer_name):
    """
    :param model: model for classification
    :param layer_name: name of the layer which will be frozen
    :return: stops the counting of the gradients besides the inputted layers
    """
    requires_grad = False
    for name, params in model.named_parameters():
        if layer_name in name:
            requires_grad = True

        params.requires_grad = requires_grad


# function for model choosing
def initialize_model(model_name, num_classes, use_pretrained=True, freeze=False):
    """
    :param model_name: name of classification model
    :param num_classes: number of output classes
    :param use_pretrained: download  pretrained-True/False
    :param freeze: freeze layers until last layer - True/False
    :return: chosen model, input size for resizing in transformation part
    """

    chosen_model = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18 """

        chosen_model = models.resnet18(pretrained=use_pretrained)
        if freeze:
            freeze_until(chosen_model, 'layer4')
        num_ftrs = chosen_model.fc.in_features
        chosen_model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = (224, 224)

    elif model_name == "resnet34":
        """ Resnet34 """

        chosen_model = models.resnet34(pretrained=use_pretrained)
        if freeze:
            freeze_until(chosen_model, 'layer4')
        num_ftrs = chosen_model.fc.in_features
        chosen_model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = (224, 224)

    elif model_name == "alexnet":
        """ AlexNet """

        chosen_model = models.alexnet(pretrained=use_pretrained)
        if freeze:
            freeze_until(chosen_model, 'classifier[6]')
        chosen_model.classifier[4] = nn.Linear(4096, 1024)
        chosen_model.classifier[6] = nn.Linear(1024, num_classes)
        input_size = (224, 224)

    elif model_name == 'cnn':
        """ Implemented CNN """
        chosen_model = CNN(num_classes)
        input_size = (448, 448)
    else:
        print("Invalid model name, exiting...")
        exit()

    return chosen_model, input_size


# get transformed dataloader
def get_transformed_dataloader(train_path, test_path, input_resize):
    """
    :param train_path: train folder path
    :param test_path: test folder path
    :return: transformed test and train dataloaders in dictionary
    """

    # make transformation
    chosen_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_resize),
            transforms.RandomRotation(degrees=25),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),

        'test': transforms.Compose([
            transforms.Resize(input_resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # train and test DataLoaders
    train_loader = DataLoader(
        ImageFolder(train_path, transform=chosen_transforms['train']), batch_size=8, shuffle=True, num_workers=4)

    test_loader = DataLoader(
        ImageFolder(test_path, transform=chosen_transforms['test']), batch_size=8, num_workers=4)

    data_loader = {'train': train_loader,
                   'test': test_loader}

    return data_loader


# visualize images from the dataloader batch by batch
def batch_show(data_loader, train_test='train'):
    """
    :param data_loader: dataloader
    :param train_test:str --> chose dataloader for test or train, default is train
    :return: images for one batch
    """
    batch = next(iter(data_loader[train_test]))
    batch_grid = torchvision.utils.make_grid(batch[0], nrow=10)
    plt.figure(figsize=(15, 15))
    plt.imshow(batch_grid.permute(1, 2, 0))
    plt.show()


# function for calculating amount of correct answers
def get_num_correct(pred, label):
    """
    params: prediction values, labels
    returns: amount of correct predicted values
    """
    return pred.argmax(dim=1).eq(label).sum().item()


# function for training and evaluating the image classification model
def model_train(model, optimizer, loss_function, data_loader, system_device,
                train_cnt, test_cnt, checkpoint_path, num_epochs):
    """
    :param model: CNN model
    :param optimizer: optimization algorithm
    :param loss_function: loss function from Pytorch nn library
    :param data_loader: transformed data loaders dictionary for train and test
    :param system_device: for checking device - cpu/gpu
    :param num_epochs: number of iterations
    :param train_cnt: count of samples of train dataset
    :param test_cnt: count of samples of test dataset
    :param checkpoint_path: path for saving trained model with torch.save
    :return: lists or train loss, test loss, train accuracy, test accuracy
    """

    # set variables for keeping the loss and accuracy
    train_loss = []
    test_loss = []
    train_acc_lst = []
    test_acc_lst = []
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in tqdm(range(num_epochs), desc='Iterating through datasets..'):

        train_corrects = 0
        test_corrects = 0

        # set checkpoint for saving model and optimizer
        checkpoint = {'model_state': model.state_dict(),
                      'optimizer_state': optimizer.state_dict()}

        # Evaluation on training dataset
        for batch in data_loader['train']:  # get batch
            images, labels = batch

            if system_device == 'cuda':  # checking device
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            preds = model(images)  # pass batch
            loss = loss_function(preds, labels)  # calculate the loss

            optimizer.zero_grad()  # reset the gradients
            loss.backward()  # calculate the gradients
            optimizer.step()  # update parameters

            train_loss.append(loss.item())
            train_corrects += get_num_correct(preds, labels)

        train_accuracy = (train_corrects / train_cnt) * 100
        train_acc_lst.append(train_accuracy)

        # Evaluation on testing dataset
        with torch.no_grad():
            for batch in data_loader['test']:
                images, labels = batch

                if system_device == 'cuda':  # checking device
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())

                preds = model(images)
                loss = loss_function(preds, labels)
                test_loss.append(loss.item())
                test_corrects += get_num_correct(preds, labels)

        test_accuracy = (test_corrects / test_cnt) * 100
        test_acc_lst.append(test_accuracy)

        # check the best accuracy and epoch, save the model and optimizer states
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(checkpoint, checkpoint_path)  # save the checkpoint dictionary

        if (epoch + 1) % 5 == 0:
            print(f'Epoch: {epoch + 1}')
            print(f'train min loss: {np.min(train_loss):.4f}, test min loss: {np.min(test_loss):.4f}')
            print(f'train accuracy: {train_accuracy:.2f}%, test accuracy: {test_accuracy:.2f}%')
            print(f'train correct answers: {train_corrects}, test correct answers: {test_corrects}')
            print(f'Train best accuracy: {np.max(train_acc_lst):.2f}')
            print(f'Test best accuracy: {np.max(test_acc_lst):.2f}')
            print(f'Best epoch-accuracy: {best_epoch}')
            print('-' * 10)

        # check if the test accuracy and loss doesn't change in valuable amount, break the iterations
        if len(test_acc_lst) > 10:
            if best_accuracy not in test_acc_lst[-10:]:
                break
            if abs(min(test_loss) - max(test_loss)) < 0.001:
                break

    return train_loss, test_loss, train_acc_lst, test_acc_lst


"""FUNCTIONS FOR ANALYZING EVALUATION RESULTS"""

# getting predictions for the entire dataset
@torch.no_grad()
def get_all_preds_labels(network, loader, system_device):
    """
       params: model, loader
       returns: all predictions and labels
    """

    all_preds = torch.tensor([])
    all_labels = torch.tensor([])

    for batch in loader:
        images, labels = batch

        if system_device == 'cuda':   # checking device
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        preds = network(images)

        # get predictions in tensor
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )

        # get labels in tensor
        all_labels = torch.cat(
            (all_labels, labels),
            dim=0
        )

    return all_preds, all_labels


# get confusion matrix with predicted values
def get_cmt(set_labels, set_preds, classes_number):
    """
       params: train/test dataset labels, predictions, number of classes
       returns: confusion matrix
    """

    # 1. concat in one tensor the true and predicted values
    stacked = torch.stack(
        (
            set_labels,
            set_preds.argmax(dim=1)
        ),
        dim=1
    )

    stacked = stacked.int()

    # 2. for making confusion matrix, create empty tensor which will match the number of classes
    conf_matrix = torch.zeros(classes_number, classes_number, dtype=torch.int32)

    # 3. fill the empty confusion matrix with predicted and true values
    for pare in stacked:
        tl, pl = pare.tolist()
        conf_matrix[tl, pl] = conf_matrix[tl, pl] + 1

    return conf_matrix


# function for plotting the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    """Plots the confusion matrix"""

    import itertools
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=70)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

