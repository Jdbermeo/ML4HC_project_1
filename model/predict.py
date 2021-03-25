import os

import pandas as pd
import tensorflow as tf
from keras_unet.metrics import iou, iou_thresholded

from model import model_utils
from preprocessing import get_ct_scan_information
from model.loss_functions import dice_coef_loss, binary_focal_loss, jaccard_distance_loss


def predict(data_path_source_dir_: str, training_params: dict, model_params: dict) -> None:

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

    model_utils.predict_test_set(
        test_df_=x_ts_df,
        pred_dims=preprocesing_params['resize_dim'],
        test_dims=predict_params['test_dims'],
        model_=model,
        pixel_threshold=model_params['output_threshold'],
        prediction_batch_size=predict_params['prediction_batch_size'],
        output_dir=predict_params['output_dir']
    )




