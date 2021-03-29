import os

import cv2 as cv
import numpy as np

import pandas as pd
import tensorflow as tf
from keras_unet.metrics import iou, iou_thresholded

from model import img_generator
from preprocessing import get_ct_scan_information
from model.loss_functions import dice_coef_loss, binary_focal_loss, jaccard_distance_loss


def predict(data_path_source_dir_: str, training_params: dict, model_params: dict) -> None:
    """

    :param data_path_source_dir_: Path to directory where the image data is stored assuming existence of
                                    imageTr/Ts and labelTr subdirs
    :param training_params: Dictionary with parameters of the different functions used for training
    :param model_params: Dictionary with parameters of the different functions used to create and compile the Unet
    :return:
    """

    predict_params = model_params['predict_params']
    preprocesing_params = training_params['preprocesing_params']

    # Load model that will be used to predict
    model = tf.keras.models.load_model(
        model_params['best_model_path'],
        custom_objects={'iou': iou, 'iou_thresholded': iou_thresholded,
                        'binary_focal_loss_fixed': binary_focal_loss(**training_params['loss_function_params']),
                        'dice_coef_loss': dice_coef_loss,
                        'jaccard_distance_loss': jaccard_distance_loss
                        })

    # Create generator for the train set
    preprocess_object_storing_dir_ = training_params['preprocess_object_storing_dir']
    x_ts_df_path = os.path.join(preprocess_object_storing_dir_, 'x_ts_df.pkl')

    if os.path.isfile(x_ts_df_path):
        x_ts_df = pd.read_pickle(x_ts_df_path)
    else:
        _, x_ts_df = get_ct_scan_information.build_train_test_df(data_path_source_dir_)

    predict_test_set(
        test_df_=x_ts_df,
        pred_dims=preprocesing_params['resize_dim'],
        test_dims=predict_params['test_dims'],
        model_=model,
        pixel_threshold=model_params['output_threshold'],
        prediction_batch_size=predict_params['prediction_batch_size'],
        output_dir=predict_params['output_dir']
    )


def predict_test_set(test_df_: pd.DataFrame, pred_dims: tuple, test_dims: tuple,  model_, pixel_threshold: float = 0.5,
                     prediction_batch_size: int = 32, output_dir: str = 'test_pred') -> None:
    """
    loads a model spcified in `config.yml` and generates predictions for the images in `test_df` and stores
     the predictions in `.npz` format. It resizes predictions to match `test_dims`.

    :param test_df_: Dataframe with path info and index by (img_num, slice_num) of the test set
    :param pred_dims: Input size of 2D images used by the model
    :param test_dims: Dimensions of the images in the test set
    :param model_: Model to use to generate the predictions
    :param pixel_threshold: Threshold for pixel values to convert them to 0s or 1s
    :param prediction_batch_size: mini-batchsize to use when evaluating the model
    :param output_dir: Directory to store the predictions
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_dx, df_ in test_df_.groupby(level=0):
        full_img_path = df_.loc[img_dx].iloc[0]['x_ts_img_path']
        img_name = os.path.basename(full_img_path).split('.')[0]

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
