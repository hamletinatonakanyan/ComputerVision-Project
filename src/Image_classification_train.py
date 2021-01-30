# Implementation of the training process with created functions

# import the necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import Image_classification_functions as clf


# function for implementation  training process using all training created functions
def training_process(name_of_model, optimization, learning_rate, epochs_number, data_path,
                     path_saving_model, file_name_saving_model, load_saved_model=False,
                     pretrained=True, freeze_layers=False):

    # check the versions
    print(f'torch version: {torch.__version__}')
    print(f'torchvision version: {torchvision.__version__}')

    # check the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # path of the train and test datasets
    train_folder_path = f'{data_path}/train'
    test_folder_path = f'{data_path}/test'

    # get the amount of samples of train and test datasets
    train_count = clf.get_dataset_length(train_folder_path)
    test_count = clf.get_dataset_length(test_folder_path)
    print(f'Number of train samples: {train_count} \nNumber of test samples:  {test_count}')

    # get names and number of the classes
    classes, classes_amount = clf.get_classes(train_folder_path)

    # chose the model
    clf_model, input_size = clf.initialize_model(name_of_model, classes_amount, pretrained, freeze_layers)

    # get total parameters requiring grad during training process
    total_params_grad = sum(params.numel() for params in clf_model.parameters() if params.requires_grad)
    print(f'Total parameters requiring grad: {total_params_grad}')

    # get train and test dataloaders
    dataloader = clf.get_transformed_dataloader(train_folder_path, test_folder_path, input_size)

    # show one batch from the dataset
    clf.batch_show(dataloader)

    # set optimizer
    if optimization == 'adam':
        optimization = optim.Adam(filter(lambda gr: gr.requires_grad, clf_model.parameters()), lr=learning_rate)
    elif optimization == 'sgd':
        optimization = torch.optim.SGD(filter(lambda gr: gr.requires_grad, clf_model.parameters()),
                                       lr=learning_rate, momentum=0.9)

    # set the loss function
    criterion = nn.CrossEntropyLoss()

    # set path for saving model,
    saved_model_path = f'{path_saving_model}/{file_name_saving_model}'

    # load saved model if parameter set True(if the training process was interrupted and there is need to continue)
    if load_saved_model:
        checkpoint = torch.load(saved_model_path)
        clf_model.load_state_dict(checkpoint['model_state'])
        optimization.load_state_dict(checkpoint['optimizer_state'])
        clf_model.eval()

    # train the model
    train_loss, test_loss, train_acc, test_acc = clf.model_train(clf_model, optimization,
                                                                 criterion, dataloader, device,
                                                                 train_count, test_count,
                                                                 saved_model_path,
                                                                 num_epochs=epochs_number)

    """ ANALYZING, PLOTTING THE RESULTS"""

    # Plotting the train and test losses
    epochs = [i for i in range(1, 50)]
    train_test_loss_df = pd.DataFrame((zip(train_loss, test_loss, epochs)),
                                      columns=['train_loss', 'test_loss', 'epochs'])
    df_loss_plot = train_test_loss_df.pivot_table(index='epochs', values=['train_loss', 'test_loss'])
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_loss_plot)
    plt.show()

    # Plotting the train and test accuracies
    epoch = [i for i in range(1, 50)]
    train_test_acc_df = pd.DataFrame((zip(train_acc, test_acc, epoch)),
                                     columns=['train_accuracy', 'test_accuracy', 'epochs'])
    df_acc_plot = train_test_acc_df.pivot_table(index='epochs', values=['train_accuracy', 'test_accuracy'])
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_acc_plot)
    plt.show()

    # Train dataset confusion matrix
    train_preds, train_labels = clf.get_all_preds_labels(clf_model, dataloader['train'], device)
    train_conf_matx = clf.get_cmt(train_labels, train_preds, classes_amount)
    clf.plot_confusion_matrix(train_conf_matx, classes, title='CNN train dataset confusion matrix')

    # Test dataset confusion matrix
    test_preds, test_labels = clf.get_all_preds_labels(clf_model, dataloader['test'], device)
    test_conf_matx = clf.get_cmt(test_labels, test_preds, classes_amount)
    clf.plot_confusion_matrix(test_conf_matx, classes, title='CNN test dataset confusion matrix')


if __name__ == '__main__':
    print('CNN classifier training module: Running directly')

