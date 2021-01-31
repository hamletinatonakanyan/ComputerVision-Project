# import the necessary libraries
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import Image_classification_functions as clf


# Functions for creating prediction process CNN classification model

# load the saved model
def load_checkpoint(filepath, model):
    """
    :param filepath: path of the saved model
    :param model: chosen model which was saved
    :return: saved model_state
    """

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model


# get transformed dataloader
def get_dataloader(folder_path, input_resize):
    """
    :param folder_path: folder path of the data
    :param input_resize: image size for the input to the chosen model
    :return: transformed dataloader
    """
    pred_dataset = transforms.Compose([
        transforms.Resize(input_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataloader = DataLoader(
        ImageFolder(folder_path, transform=pred_dataset), batch_size=8, num_workers=4)

    return dataloader


# get accuracy
def get_accuracy(transformed_dataloader, model, count_dataset, system_device):
    """
    :param loader: transformed data loader
    :param count_dataset: number of dataset's samples
    :param model: chosen model
    :param system_device: cpu/gpu
    :return: prediction's accuracy
    """

    correct_values = 0

    with torch.no_grad():
        for batch in transformed_dataloader:
            images, labels = batch

            if system_device == 'cuda':  # checking device
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            preds = model(images)
            correct_values += clf.get_num_correct(preds, labels)

    accuracy = (correct_values / count_dataset) * 100

    return accuracy


# Implementation of the prediction process
def prediction_process(name_of_model, folder_path, saved_model_path, saved_model_file,
                       model_pretrained=True, freeze_model_layers=False):

    # check the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get length of the dataset
    dataset_length = clf.get_dataset_length(folder_path)

    # get names and number of the classes
    classes, classes_amount = clf.get_classes(folder_path)

    # chose the model
    clf_model, input_size = clf.initialize_model(name_of_model, classes_amount,
                                                 model_pretrained, freeze_model_layers)

    # get transformed data loader
    dataloader = get_dataloader(folder_path, input_size)

    # set path for saving model,
    saved_model_file_path = f'{saved_model_path}/{saved_model_file}'

    # get the state of the saved model
    prediction_model = load_checkpoint(saved_model_file_path, clf_model)

    # get the prediction accuracy
    pred_accuracy = get_accuracy(dataloader, prediction_model, dataset_length, device)

    print(f'Prediction accuracy: {pred_accuracy:.2f}%')

    # get Confusion Matrix for predictions
    predictions, labels = clf.get_all_preds_labels(prediction_model, dataloader, device)
    preds_conf_matx = clf.get_cmt(labels, predictions, classes_amount)
    clf.plot_confusion_matrix(preds_conf_matx, classes, title='CNN prediction dataset Confusion Matrix')


if __name__ == '__main__':
    print('CNN classifier prediction module: Running directly')
