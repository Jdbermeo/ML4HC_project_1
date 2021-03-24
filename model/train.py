import os
import logging

import tensorflow as tf

from model import training_utils, img_generator, model_utils
from preprocessing import get_ct_scan_information


def main(data_path_source_dir_: str, object_storing_dir_: str, preprocesing_params_: dict,
         sampling_params_: dict, augmentation_params_: dict, training_params: dict,
         folds: int = 10, seed: int = 123):

    os.makedirs(object_storing_dir_, exist_ok=True)

    # Create dataframes in the format and with the information required by the generators that will feed the model
    tr_df, x_ts_df = get_ct_scan_information.build_train_test_df(data_path_source_dir_)
    cancer_pixels_df = get_ct_scan_information.get_cancer_pixel_count_df(full_tr_df=tr_df)

    # Create CV folds for `tr_df`
    tr_fold_df_dict = training_utils.generate_fold_dict(df_=tr_df, n_folds=folds, seed=seed)

    # Take one of the folds to train the model
    tr_fold_0_df = tr_fold_df_dict['fold_0']['train']
    holdout_fold_0_df = tr_fold_df_dict['fold_0']['holdout']
    logging.info(f'Rows in the train set in each fold (before sampling): {tr_fold_0_df.shape[0]}')
    logging.info(f'Rows in the holdout set in each fold (before sampling): {holdout_fold_0_df.shape[0]}')

    # Let's add the information of which slices contain cancer and which do not
    tr_fold_0_df_cancer_info = get_ct_scan_information.add_cancer_pixel_info(
        df_=tr_fold_0_df.copy(),
        cancer_pixels_df_=cancer_pixels_df
    )

    holdout_fold_0_df_cancer_info = get_ct_scan_information.add_cancer_pixel_info(
        df_=holdout_fold_0_df.copy(),
        cancer_pixels_df_=cancer_pixels_df
    )

    # Let's create a generator for the train and holdout set using the first fold
    train_data_generator = img_generator.DataGenerator2D(
        df=tr_fold_0_df_cancer_info, x_col='x_tr_img_path', y_col='y_tr_img_path',
        shuffle=True, shuffle_depths=True,
        batch_size=training_params['batch_size_train'],
        class_sampling=sampling_params_['class_sampling'], depth_class_col=sampling_params_['depth_class_col'],
        resize_dim=preprocesing_params_['resize_dim_'], hounsfield_min=preprocesing_params_['hounsfield_min'],
        hounsfield_max=preprocesing_params_['hounsfield_max'],
        rotate_range=augmentation_params_['rotate_range'], horizontal_flip=augmentation_params_['horizontal_flip'],
        vertical_flip=augmentation_params_['vertical_flip'], random_crop=augmentation_params_['random_crop'],
        shearing=augmentation_params_['shearing'], gaussian_blur=augmentation_params_['gaussian_blur']
    )

    holdout_data_generator = img_generator.DataGenerator2D(
        df=holdout_fold_0_df, x_col='x_tr_img_path', y_col='y_tr_img_path',
        shuffle=False,
        batch_size=training_params['batch_size_val'],
        resize_dim=preprocesing_params_['resize_dim_'], hounsfield_min=preprocesing_params_['hounsfield_min'],
        hounsfield_max=preprocesing_params_['hounsfield_max']
    )

    # Build model
    model = model_utils.create_model(
        resize_dim_=preprocesing_params_['resize_dim'],
        lr_=training_params['learning_rate'],
        loss_function_name_=training_params['loss_function_name'],
        object_storing_dir_=training_params['object_storing_dir'],
        num_filters_first_level_=32, gamma=2., alpha=0.7
    )

    # Set the callbacks
    my_callbacks = [
        tf.keras.callbacks.LearningRateScheduler(training_utils.scheduler, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'./{object_storing_dir_}' + '/model_sampling.{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=f'./{object_storing_dir_}/logs'),
    ]

    # Train the model
    model.fit(train_data_generator, validation_data=holdout_data_generator,
               epochs=training_params['num_epoch'], callbacks=my_callbacks)

    model.save(f'./{object_storing_dir_}/end_of_training_version')
