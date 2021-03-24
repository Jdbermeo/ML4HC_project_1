import os
from typing import Tuple

import cv2 as cv
import pandas as pd
import numpy as np
from keras_unet.models import custom_unet
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded

import training_utils
import img_generator


def create_model(resize_dim_: Tuple[int, int], lr_: float, loss_function_name_: str,
                 object_storing_dir_: str, num_filters_first_level_: int = 32, num_classes_: int = 1,
                 dropout_rate_: float = 0.2, use_batch_norm_: bool = True, **loss_kwargs):


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
        # optimizer=SGD(lr=0.01, momentum=0.99),
        # loss='binary_crossentropy',
        loss=loss_function,
        metrics=[iou, iou_thresholded]
    )

    return model


def predict_test_set(test_df_: pd.DataFrame, pred_dims: tuple, test_dims: tuple,  model_, pixel_threshold: float = 0.5,
                     prediction_batch_size: int = 32, output_dir: str = 'test_pred') -> None:
    """

    :param test_df_:
    :param pred_dims:
    :param test_dims:
    :param model_:
    :param pixel_threshold:
    :param prediction_batch_size:
    :param output_dir:
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_dx, df_ in test_df_.groupby(level=0):
        full_img_path = df_.loc[img_dx].iloc[0]['x_ts_img_path']
        img_name = full_img_path.split('/')[-1].split('.')[0]

        img_i_generator = img_generator.DataGenerator2D(
            df=df_, x_col='x_ts_img_path', y_col=None,
            batch_size=prediction_batch_size, num_classes=None, shuffle=False,
            resize_dim=pred_dims)

        # Predict for a group of cuts of the same image
        for i, (X_cut_i, _) in enumerate(img_i_generator):
            y_cut_i_predict = model_.predict(X_cut_i)

            # Resize prediction to match label mask dimensions and restack
            #  the predictions so that hey are channel last
            for j, depth_i in enumerate(range(X_cut_i.shape[0])):
                y_cut_i_predict_resized_j = cv.resize(
                    y_cut_i_predict[j, :, :], test_dims,
                    interpolation=cv.INTER_CUBIC)  # INTER_LINEAR is faster but INTER_CUBIC is better

                # Add extra dim at the end
                y_cut_i_predict_resized_j = y_cut_i_predict_resized_j.reshape(y_cut_i_predict_resized_j.shape + (1,))

                if j == 0:
                    y_cut_i_predict_resized = y_cut_i_predict_resized_j

                else:
                    y_cut_i_predict_resized = np.concatenate([y_cut_i_predict_resized, y_cut_i_predict_resized_j],
                                                             axis=2)

            # When there is only one image in the minibatch it adds an extra dimension
            if len(y_cut_i_predict_resized.shape) > 3:
                y_cut_i_predict_resized = np.squeeze(y_cut_i_predict_resized, axis=3)

            # Now stack the minibatches along the 3rd axis to complete the 3D image
            if i == 0:
                y_i_predict_3d = y_cut_i_predict_resized

            else:
                y_i_predict_3d = np.concatenate([y_i_predict_3d, y_cut_i_predict_resized], axis=2)

            y_i_predict_3d_thres = (y_i_predict_3d > pixel_threshold) * 1

            np.savez(os.path.join(output_dir, f'{img_name}_pred.npz'),
                     y_i_predict_3d_thres)
