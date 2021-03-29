import os
from typing import Tuple

from keras_unet.models import custom_unet
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded

from model import training_utils


def create_model(resize_dim_: Tuple[int, int], lr_: float, loss_function_name_: str,
                 object_storing_dir_: str, num_filters_first_level_: int = 32, num_classes_: int = 1,
                 dropout_rate_: float = 0.2, use_batch_norm_: bool = True, **loss_kwargs):
    """
    Create 2D Unet model that will be used to predict segmentation masks on single slices of the 3D scan.

    Returns a keras model that predicts single channel masks on single channel 2D images

    :param resize_dim_: 2D dimensions of the images that the model cn predict on.
    :param lr_: learning rate
    :param loss_function_name_: name of loss function to use
    :param object_storing_dir_: Directory where model objects will be stored. In this case, the model's summary.
    :param num_filters_first_level_:
    :param num_classes_:
    :param dropout_rate_:
    :param use_batch_norm_:
    :param loss_kwargs: In case the loss function takes arguments, add them as a dictionary

    :return:
    """

    # Build model with `custom_unet` library
    model = custom_unet(
        input_shape=resize_dim_ + (1,),
        use_batch_norm=use_batch_norm_,
        num_classes=num_classes_,
        filters=num_filters_first_level_,
        dropout=dropout_rate_,
        output_activation='sigmoid')

    with open(os.path.join(object_storing_dir_, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Get loss function to use
    loss_function = training_utils.get_loss_function(loss_function_name_, **loss_kwargs)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=lr_),
        loss=loss_function,
        metrics=[iou, iou_thresholded]
    )

    return model
