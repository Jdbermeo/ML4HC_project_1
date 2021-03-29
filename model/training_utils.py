import os
import logging
from typing import Dict, Union, Callable, Tuple

import pandas as pd
from sklearn.model_selection import KFold

from model.loss_functions import jaccard_distance_loss, binary_focal_loss, dice_coef_loss
from model import img_generator
from model.img_generator import DataGenerator2D
from preprocessing import get_ct_scan_information


def scheduler(epoch, lr):
    """
    Schedule of the learning rate during training based on the epoch number. Currently it is manually set for each
    epoch range instead of it being dependent on the current learning rate value
    :param epoch:
    :param lr:
    :return:
    """

    if epoch <= 12:
        return 1e-3

    elif 12 < epoch <= 25:
        return 1e-4

    else:
        return 1e-5


def generate_fold_dict(df_: pd.DataFrame, n_folds: int = 3, seed: int = 123) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create the folds that could be used for cross validation

    :param df_: tr_df with cancer pixel information
    :param n_folds:
    :param seed:
    :return:
    """

    img_num_idx_list = df_.index.levels[0]
    folder = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    df_fold_dict = dict()

    for i, (train_fold_i, holdout_i) in enumerate(folder.split(img_num_idx_list)):
        train_fold_i_idx = img_num_idx_list[train_fold_i]
        holdout_i_idx = img_num_idx_list[holdout_i]

        df_fold_dict[f'fold_{i}'] = {
            'train': df_.loc[pd.IndexSlice[train_fold_i_idx, :], :],
            'holdout': df_.loc[pd.IndexSlice[holdout_i_idx, :], :]
        }

    return df_fold_dict


def get_loss_function(loss_function_name: str, **kwargs) -> Union[str, Callable]:
    """
    Return one of the custom loss functions used for image segmentation

    :param loss_function_name: String with name of los function to use
    :param kwargs: Dictionary of arguments to create the specific loss function (required for binary_focal_loss)
    :return: The custom function or name of an already implemented loss function in keras
    """
    if loss_function_name == 'jaccard_loss':
        return jaccard_distance_loss

    elif loss_function_name == 'dice_loss':
        return dice_coef_loss

    elif loss_function_name == 'binary_focal_loss':
        return binary_focal_loss(**kwargs)

    elif loss_function_name == 'binary_crossentropy':
        return 'binary_crossentropy'

    else:
        raise Exception('Loss function not included in `get_loss_function()`')


def prepare_train_holdout_generators(data_path_source_dir_: str, training_params: dict) -> \
        Tuple[DataGenerator2D, DataGenerator2D, pd.DataFrame, pd.DataFrame]:
    """
    Takes directory where CT scan data is stored. Assumes subdirs imagesTs/Tr and labelsTr structure. Returns
    generators that read the scan data by 2D slices and feed them to the model with an augmentation and class
    balancing logic through up and down sampling.

    :param data_path_source_dir_: Dir where CT scan data is stored. Assumes subdirs imagesTs/Tr and labelsTr
    :param training_params: Training parameters from the YAML
    :return: train_data_generator, holdout_data_generator objects to feed the Keras model
    """
    sampling_params = training_params['sampling_params']
    augmentation_params = training_params['augmentation_params']
    preprocesing_params = training_params['preprocesing_params']
    preprocess_object_storing_dir = training_params['preprocess_object_storing_dir']

    os.makedirs(preprocess_object_storing_dir, exist_ok=True)
    x_ts_df_path = os.path.join(preprocess_object_storing_dir, 'x_ts_df.pkl')
    tr_fold_0_df_cancer_info_path = os.path.join(preprocess_object_storing_dir, 'tr_fold_0_df_cancer_info.pkl')
    holdout_fold_0_df_cancer_info_path = os.path.join(preprocess_object_storing_dir,
                                                      'holdout_fold_0_df_cancer_info.pkl')

    if os.path.isfile(tr_fold_0_df_cancer_info_path) and os.path.isfile(holdout_fold_0_df_cancer_info_path):
        tr_fold_0_df_cancer_info = pd.read_pickle(tr_fold_0_df_cancer_info_path)
        holdout_fold_0_df_cancer_info = pd.read_pickle(holdout_fold_0_df_cancer_info_path)

    else:
        # Create dataframes in the format and with the information required by the generators that will feed the model
        logging.info("Create dataframes in the format and with the information required by the generators that will "
                     "feed the model")
        tr_df, x_ts_df = get_ct_scan_information.build_train_test_df(data_path_source_dir_)

        logging.info('Store x_ts_df_path to use in prediction')
        x_ts_df.to_pickle(x_ts_df_path)

        logging.info("Get information on which slices are labeled with cancer")
        cancer_pixels_df_path = os.path.join(preprocess_object_storing_dir, 'cancer_pixels_df.pkl')

        if os.path.isfile(cancer_pixels_df_path):
            logging.info("Read cancer_pixels_df from pickle")
            cancer_pixels_df = pd.read_pickle(cancer_pixels_df_path)

        else:
            logging.info("Create cancer_pixels_df and save it into a pkl")
            cancer_pixels_df = get_ct_scan_information.get_cancer_pixel_count_df(full_tr_df=tr_df)
            cancer_pixels_df.to_pickle(cancer_pixels_df_path)

        # Create CV folds for `tr_df`
        tr_fold_df_dict = generate_fold_dict(df_=tr_df, n_folds=training_params['folds'],
                                             seed=training_params['seed'])

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

        tr_fold_0_df_cancer_info.to_pickle(tr_fold_0_df_cancer_info_path)

        holdout_fold_0_df_cancer_info = get_ct_scan_information.add_cancer_pixel_info(
            df_=holdout_fold_0_df.copy(),
            cancer_pixels_df_=cancer_pixels_df
        )

        holdout_fold_0_df_cancer_info.to_pickle(
            os.path.join(preprocess_object_storing_dir, 'holdout_fold_0_df_cancer_info.pkl')
        )

    # Let's create a generator for the train and holdout set using the first fold
    train_data_generator = img_generator.DataGenerator2D(
        df=tr_fold_0_df_cancer_info, x_col='x_tr_img_path', y_col='y_tr_img_path',
        shuffle=True, shuffle_depths=True,
        batch_size=training_params['batch_size_train'],
        class_sampling=sampling_params['class_sampling'], depth_class_col=sampling_params['depth_class_col'],
        resize_dim=preprocesing_params['resize_dim'], hounsfield_min=preprocesing_params['hounsfield_min'],
        hounsfield_max=preprocesing_params['hounsfield_max'],
        rotate_range=augmentation_params['rotate_range'], horizontal_flip=augmentation_params['horizontal_flip'],
        vertical_flip=augmentation_params['vertical_flip'], random_crop=augmentation_params['random_crop'],
        shearing=augmentation_params['shearing'], gaussian_blur=augmentation_params['gaussian_blur']
    )

    holdout_data_generator = img_generator.DataGenerator2D(
        df=holdout_fold_0_df_cancer_info, x_col='x_tr_img_path', y_col='y_tr_img_path',
        shuffle=False,
        batch_size=training_params['batch_size_val'],
        resize_dim=preprocesing_params['resize_dim'], hounsfield_min=preprocesing_params['hounsfield_min'],
        hounsfield_max=preprocesing_params['hounsfield_max']
    )

    return train_data_generator, holdout_data_generator, tr_fold_0_df_cancer_info, holdout_fold_0_df_cancer_info
