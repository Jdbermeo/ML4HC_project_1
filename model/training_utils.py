from typing import Dict, Union, Callable

import pandas as pd
from sklearn.model_selection import KFold

from loss_functions import jaccard_distance_loss, binary_focal_loss, dice_coef_loss


def scheduler(epoch, lr):
    if epoch <= 3:
        return 1e-2

    elif 3 < epoch <= 12:
        return 1e-3

    elif 12 < epoch <= 25:
        return 1e-4

    else:
        return 1e-5


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


def get_loss_function(loss_function_name: str, **kwargs) -> Union[str, Callable]:

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