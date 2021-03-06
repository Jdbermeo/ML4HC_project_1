import os
import random
import logging

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import training_utils, model_utils, metric_utils


def train(data_path_source_dir_: str, training_params: dict, model_params: dict):
    """
    Sequence all the steps in the helper scripts in accordance to the parameter values given to do the following:
        - Create generator objects that can feed 2D single channel images to a keras.model fit() method
        - Create a 2D Unet
        - Fit the Unet and store the information of its progress and best performing versions
        - Obtain IoU on train and holdout
        - Store information about the pixel values predicted by the model on a sample of 5 images

    :param data_path_source_dir_: Path to directory where the image data is stored assuming existence of
                                    imageTr/Ts and labelTr subdirs
    :param training_params: Dictionary with parameters of the different functions used for training
    :param model_params: Dictionary with parameters of the different functions used to create and compile the Unet

    :return:
    """
    logging.info('-------------------TRAIN-------------------')
    model_object_storing_dir = training_params['model_object_storing_dir']
    os.makedirs(model_object_storing_dir, exist_ok=True)

    preprocesing_params_: dict = training_params["preprocesing_params"]

    # Get data generators for train and holdout
    logging.info('Get data generators for train and holdout')
    train_data_generator, holdout_data_generator, tr_fold_0_df_cancer_info, holdout_fold_0_df_cancer_info = \
        training_utils.prepare_train_holdout_generators(
            training_params=training_params, data_path_source_dir_=data_path_source_dir_)

    # Build model
    logging.info('Build model')
    model = model_utils.create_model(
        resize_dim_=preprocesing_params_['resize_dim'],
        lr_=training_params['learning_rate'],
        loss_function_name_=training_params['loss_function_name'],
        object_storing_dir_=model_object_storing_dir,
        num_filters_first_level_=training_params['num_filters_first_level'],
        **training_params['loss_function_params']
    )

    # Set the callbacks
    my_callbacks = [
        tf.keras.callbacks.LearningRateScheduler(training_utils.scheduler, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'./{model_object_storing_dir}' + '/model.{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=f'./{model_object_storing_dir}/logs'),
    ]

    # Train the model
    logging.info('Train the model')
    model.fit(train_data_generator, validation_data=holdout_data_generator,
              epochs=training_params['num_epoch'], callbacks=my_callbacks)

    # Save the models training history
    logging.info("Save the model's training history")
    pd.DataFrame(model.history.history).to_csv(os.path.join(model_object_storing_dir, 'training_history.csv'))

    # Store last version of the model
    logging.info("Store last version of the model")
    model.save(f'./{model_object_storing_dir}/end_of_training_version')

    # Store metrics for the train and holdout sets
    logging.info('Store metrics for the train and holdout sets')
    store_metrics(df_cancer_info=tr_fold_0_df_cancer_info, dataset_type='train', model=model,
                  model_object_storing_dir=model_object_storing_dir, model_params=model_params,
                  preprocesing_params_=preprocesing_params_)

    store_metrics(df_cancer_info=holdout_fold_0_df_cancer_info, dataset_type='holdout', model=model,
                  model_object_storing_dir=model_object_storing_dir, model_params=model_params,
                  preprocesing_params_=preprocesing_params_)


def store_metrics(df_cancer_info: pd.DataFrame, dataset_type: str, model: tf.keras.Model, model_object_storing_dir: str,
                  model_params: dict, preprocesing_params_: dict):
    """
    Get the IoU metrics per image for the images in df_cancer_info using the model `model`. It stores the IoU for each
    3D image as well as the mean IoU over all images.

    Additionally, it also stores the min and max output pixel values as well as a histogram of unique pixel values for a
     sample of 5 predicted images. This information is to be used to set the threshold

    :param df_cancer_info:
    :param dataset_type:
    :param model:
    :param model_object_storing_dir:
    :param model_params:
    :param preprocesing_params_:
    :return:
    """
    # Store the performance on the holdout set
    iou_df, _, y_pred_list = metric_utils.calculate_iou_df(
        df_=df_cancer_info, img_dims=preprocesing_params_['resize_dim'],
        model_=model, pixel_threshold=model_params['output_threshold'],
        prediction_batch_size=model_params['predict_params']['prediction_batch_size'])

    iou_df.to_pickle(
        os.path.join(model_object_storing_dir, f'iou_per_3d_img_{dataset_type}_df.pkl')
    )

    with open(os.path.join(model_object_storing_dir, f'Training IoU in 3D over {dataset_type}.txt'), 'w') as f:
        f.write(f'Average IoU over the {dataset_type} set is: {iou_df.iou.mean()}')

    # Save a histogram of types of predicted pixel values by the model as a sanity check.
    #  This will also be used to choose the bounds to tune the threshold
    os.makedirs(os.path.join(model_object_storing_dir, 'activation_values'), exist_ok=True)
    os.makedirs(os.path.join(model_object_storing_dir, 'activation_values', dataset_type), exist_ok=True)

    for i, img_i in enumerate(random.sample(y_pred_list, 5)):
        pd.Series(np.unique(img_i)).plot.hist(bins=35, density=True)
        plt.title(f'Histogram of types of predicted pixel values on {dataset_type} set in image {i}')
        plt.savefig(os.path.join(model_object_storing_dir, 'activation_values', dataset_type,
                                 f'Histogram of types of predicted pixel values img_{i} in {dataset_type}.jpg'))
