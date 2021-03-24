import os

import cv2 as cv
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import KFold

from img_generator import DataGenerator2D


def generate_fold_dict(df_: pd.DataFrame, n_folds: int = 3, seed: int = 123) -> Dict[str, Dict[str, pd.DataFrame]]:
    """

    :param df_:
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


def calculate_iou(target: np.ndarray, prediction: np.ndarray) -> float:
    """

    :param target:
    :param prediction:
    :return:
    """
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection.astype(np.float64)) / np.sum(union.astype(np.float64))

    return iou_score


def calculate_iou_holdout_set(holdout_df_: pd.DataFrame, img_dims: Tuple, model_,
                              pixel_threshold: float = 0.5, prediction_batch_size: int = 32) \
        -> Tuple[pd.DataFrame, list, list]:
    """

    :param holdout_df_:
    :param img_dims:
    :param model_:
    :param pixel_threshold:
    :param prediction_batch_size:
    :return:
    """

    iou_list = list()
    y_pred_list = list()
    y_list = list()

    for img_dx, df_ in holdout_df_.groupby(level=0):
        img_i_generator = DataGenerator2D(df=df_, x_col='x_tr_img_path', y_col=None,
                                          batch_size=prediction_batch_size, num_classes=None, shuffle=False,
                                          resize_dim=img_dims)

        label_i_generator = DataGenerator2D(df=df_, x_col='x_tr_img_path', y_col='y_tr_img_path',
                                            batch_size=prediction_batch_size, num_classes=None, shuffle=False,
                                            resize_dim=None)

        # Predict for a group of cuts of the same image
        for i, ((X_cut_i, _), (_, y_cut_i)) in enumerate(zip(img_i_generator, label_i_generator)):

            y_cut_i_predict = model_.predict(X_cut_i)

            # Resize prediction to match label mask dimensions and restack
            #  the predictions so that hey are channel last
            for j, depth_i in enumerate(range(X_cut_i.shape[0])):
                y_cut_i_predict_resized_j = cv.resize(
                    y_cut_i_predict[j, :, :], y_cut_i.shape[1:],
                    interpolation=cv.INTER_CUBIC)  # INTER_LINEAR is faster but INTER_CUBIC is better

                # Add extra dim at the end
                y_cut_i_predict_resized_j = y_cut_i_predict_resized_j.reshape(y_cut_i_predict_resized_j.shape + (1,))
                y_cut_i_j = y_cut_i[j, :, :].reshape(y_cut_i[j, :, :].shape + (1,))

                if j == 0:
                    y_cut_i_predict_resized = y_cut_i_predict_resized_j
                    y_cut_i_restacked = y_cut_i_j

                else:
                    y_cut_i_predict_resized = np.concatenate([y_cut_i_predict_resized, y_cut_i_predict_resized_j],
                                                             axis=2)
                    y_cut_i_restacked = np.concatenate([y_cut_i_restacked, y_cut_i_j], axis=2)

            # When there is only one image in the minibatch it adds an extra dimension
            if len(y_cut_i_restacked.shape) > 3:
                y_cut_i_restacked = np.squeeze(y_cut_i_restacked, axis=3)

            # Now stack the minibatches along the 3rd axis to complete the 3D image
            if i == 0:
                y_i_predict_3d = y_cut_i_predict_resized
                y_i_3d = y_cut_i_restacked

            else:
                y_i_predict_3d = np.concatenate([y_i_predict_3d, y_cut_i_predict_resized], axis=2)
                y_i_3d = np.concatenate([y_i_3d, y_cut_i_restacked], axis=2)

        y_pred_list.append(y_i_predict_3d)
        y_list.append(y_i_3d)

        # Measure IoU over entire 3D image after concatenating all of the cuts
        iou_list.append({'index': img_dx,
                         'iou': calculate_iou(target=y_i_3d, prediction=(y_i_predict_3d > pixel_threshold) * 1)})

    # Let's convert the iou to a pandas dataframe
    iou_df = pd.DataFrame(iou_list).set_index('index')

    return iou_df, y_list, y_pred_list


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

        img_i_generator = DataGenerator2D(df=df_, x_col='x_ts_img_path', y_col=None,
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



