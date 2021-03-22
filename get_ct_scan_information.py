import os

from typing import List, Tuple
import pandas as pd
import nibabel as nib


def get_img_path_df(dir_: str, col_prefix: str) -> pd.DataFrame:
    """
    Create a pandas dataframe with the paths to valid '.nii.gz' files in it

    :param dir_:
    :param col_prefix: col prefix to assign name '{col_prefix}_img_path' to the column with image paths
    :return:
    """
    img_path_list = [os.path.join(dir_, filename) for filename in os.listdir(dir_) if
                     filename != '.DS_Store' and '._' not in filename]
    img_num_list = [filename.split('colon_')[1].split('.nii.gz')[0] for filename in os.listdir(dir_) if
                    filename != '.DS_Store' and '._' not in filename]

    img_path_df = pd.DataFrame({f'{col_prefix}_img_path': img_path_list,
                                'index': img_num_list}) \
        .set_index('index')

    return img_path_df


def add_depth_image(df_: pd.DataFrame, col_name: str) -> List:
    """
    Return a list with the depth of each of the image paths listed in `col_name`

    :param df_:
    :param col_name:
    :return:
    """

    channel_number_list = list()

    for index, img_path in df_[col_name].iteritems():
        channel_number_list.append(nib.load(img_path).shape[-1])

    return channel_number_list


def create_depth_based_index(df_: pd.DataFrame, col_to_use: str = 'depth') -> pd.DataFrame:
    """
    Create a new set of rows and indexes where we have a row for each possible image and depth/channel/cut it has
    :param df_:
    :param col_to_use:
    :return:
    """
    df_ = pd.DataFrame(df_[col_to_use].map(lambda depth: list(range(depth))).explode().rename('depth_i'))\
            .join(df_) \
            .set_index('depth_i', append=True)

    return df_


def build_train_test_df(data_path_source_dir_: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use the functions above to create a dataset for the train and test sets where we have as indexes the number of the
    image, each channel or depth it has, and the path to it.

    For train we generate a column for the path of the image and to the labels
    For train we only generate a column for the path of the images

    :param data_path_source_dir_:
    :return:
    """

    x_dir_path_tr = os.path.join(data_path_source_dir_, 'imagesTr')
    y_dir_path_tr = os.path.join(data_path_source_dir_, 'labelsTr')

    x_tr_df = get_img_path_df(dir_=x_dir_path_tr, col_prefix='x_tr')
    x_tr_df['x_tr_img_depth'] = add_depth_image(df_=x_tr_df, col_name='x_tr_img_path')

    y_tr_df = get_img_path_df(dir_=y_dir_path_tr, col_prefix='y_tr')
    y_tr_df['y_tr_img_depth'] = add_depth_image(df_=y_tr_df, col_name='y_tr_img_path')

    tr_df_ = x_tr_df.join(y_tr_df, how='inner')

    assert (tr_df_.x_tr_img_depth == tr_df_.y_tr_img_depth).all()

    tr_df_ = tr_df_.drop('y_tr_img_depth', axis=1).rename(columns={'x_tr_img_depth': 'depth'})

    tr_df_ = create_depth_based_index(df_=tr_df_, col_to_use='depth')

    # Convert to series
    x_dir_path_ts = os.path.join(data_path_source_dir_, 'imagesTs')
    x_ts_df_ = get_img_path_df(dir_=x_dir_path_ts, col_prefix='x_ts')
    x_ts_df_['depth'] = add_depth_image(df_=x_ts_df_, col_name='x_ts_img_path')
    x_ts_df_ = create_depth_based_index(df_=x_ts_df_, col_to_use='depth')

    return tr_df_, x_ts_df_


def add_cancer_pixel_info(df_: pd.DataFrame, cancer_pixels_df_: pd.DataFrame) -> pd.DataFrame:
    """
    Adds information of which slices contain pixels labeled as cancerous tissue assuming both `df_` and
    `cancer_pixels_df_` are indexed by image_number-depth_i and that `cancer_pixels_df_` contains the area count of
    pixels labeled as cancerous for the slices that have them.

    :param df_:
    :param cancer_pixels_df_:
    :return: df_ with 'has_cancer_pixels' and 'cancer_pixel_area' columns
    """
    df_ = df_.join(cancer_pixels_df_.set_index(['index', 'depth_i'])[['cancer_pixel_area']], how='left')
    df_['has_cancer_pixels'] = ~df_.cancer_pixel_area.isna()
    df_.cancer_pixel_area.fillna(0, inplace=True)

    return df_
