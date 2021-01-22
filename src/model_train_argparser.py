# argument parser file for CNN classification model

import argparse
import Image_classification_train as clf_nn


def main_function():
    parser = argparse.ArgumentParser(description='CNN classification model')

    parser.add_argument('-m', '--model',
                        choices=('resnet18', 'resnet34', 'alexnet', 'cnn'),
                        metavar='', required=True,
                        help='Name of the model for image classification.')

    parser.add_argument('-op', '--optimization',
                        choices=('adam', 'sgd'),
                        metavar='', required=True,
                        help='Name of the optimizer for the chosen model.')

    parser.add_argument('-l', '--learning_rate', type=float,
                        metavar='', required=True,
                        help='The learning rate for the optimizer.')

    parser.add_argument('-e', '--epochs_number', type=int,
                        choices=(5, 10, 20, 50, 60, 80, 100),
                        metavar='', required=True,
                        help='Number of epochs for iterating the model.')

    parser.add_argument('-p', '--model_pretrained', type=bool,
                        default=True,
                        metavar='', required=True,
                        help='Download model in pretrained mood: True/False.')

    parser.add_argument('-fl', '--freeze_layers', type=bool,
                        default=False,
                        metavar='', required=True,
                        help='Freeze chosen model\'s layers until last layer: True/False.')

    args = parser.parse_args()
    result = clf_nn.training_process(args.model, args.optimization, args.learning_rate,
                                     args.epochs_number, args.model_pretrained, args.freeze_layers)

    return result


if __name__ == '__main__':
    print(f'Argument parser file --> module\'s name: {__name__}')
    main_function()
