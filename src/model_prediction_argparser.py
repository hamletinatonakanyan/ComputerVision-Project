# argument parser file for the prediction process

import argparse
import Image_prediction as img_pred


def main_func():

    pred_parser = argparse.ArgumentParser(description='CNN classification model prediction process')

    pred_parser.add_argument('-mn', '--model_name',
                             choices=('resnet18', 'resnet34', 'alexnet', 'cnn'),
                             metavar='', required=True,
                             help='Name of the model for images prediction.')

    pred_parser.add_argument('-dfp', '--data_folder_path',
                             metavar='', required=True,
                             help='Data folder path of the images for prediction process')

    pred_parser.add_argument('-psm', '--path_saved_model_file',
                             metavar='', required=True,
                             help='File path of the saved model state through torch.save')

    pred_parser.add_argument('-nsm', '--name_saved_model_file',
                             metavar='', required=True,
                             help='File name with extension of the saved model\'s state through torch.save')

    pred_parser.add_argument('-mp', '--model_pretrained', type=bool,
                             default=True,
                             metavar='', required=True,
                             help='Model\'s pretrained mood: True/False. Choose the same as while train process.')

    pred_parser.add_argument('-fl', '--freeze_layers', type=bool,
                             default=False,
                             metavar='', required=True,
                             help='Freeze model\'s layers: True/False. Choose the same as while train process.')

    p_args = pred_parser.parse_args()
    pred_result = img_pred.prediction_process(p_args.model_name, p_args.data_folder_path,
                                              p_args.path_saved_model_file, p_args.name_saved_model_file,
                                              p_args.model_pretrained, p_args.freeze_layers)

    return pred_result


if __name__ == '__main__':
    print(f'Argument parser file for prediction process --> module\'s name: {__name__}')
    main_func()
