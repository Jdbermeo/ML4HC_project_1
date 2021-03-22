#!pip install keras_unet
#pip install nibabel
#!pip install focal-loss

import os
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras_unet.models import custom_unet
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded

import img_generator
import model_utils
import loss_functions

"""
Script's parameters

data_path_source_dir = os.path.join('ml4h_proj1_colon_cancer_ct')
n_folds=10
seed=123
resize_dim = (128, 128) # For the model's resizing and creation

num_epoch = 20
version = 1
split = '90_10_split'
lr = 1e-2
loss_used = 'binary_focal_loss'
augmentations = 'flips_rot_crop'
depth_shuffle = 'depth_shuffle'
imbalance_sampling = 'up_down_sampling'
attepmt_name_dir = f'v{version}_{num_epoch}_lr_{lr}_epochs_{split}_{loss_used}_{augmentations}_{depth_shuffle}_{imbalance_sampling}' 
attepmt_name_dir = os.path.join('training_runs', 'juan', 'binary_loss_only', attepmt_name_dir)

"""

# Create dataframes in the format and with the information required by the generators that will feed the model
## Each dataframes will contain the paths and depths of the images

data_path_source_dir = os.path.join('ml4h_proj1_colon_cancer_ct')

tr_df, x_ts_df = img_generator.build_train_test_df(data_path_source_dir)


"""### Create CV folds for `tr_df`

let's go for 10 folds to have a 90/10 split. We can still only use only 3 or 5 to estimate the metrics
"""

tr_fold_df_dict =  model_utils.generate_fold_dict(df_=tr_df, n_folds=10, seed=123) # 10 folds to have 90/10 split, but we can use only 3 or 5 to estimate the metrics

# Let's get the data of the first fold
tr_fold_0_df = tr_fold_df_dict['fold_0']['train']
holdout_fold_0_df = tr_fold_df_dict['fold_0']['holdout']
resize_dim = (128, 128)
logging.info(f'Rows in the train set in each fold (before sampling): {tr_fold_0_df.shape[0]}')
logging.info(f'Rows in the holdout set in each fold (before sampling): {holdout_fold_0_df.shape[0]}')

# Let's add the information of which slices contain cancer and which do not
# TODO: Add function that determines which images have cancer or not
cancer_pixels_df = pd.read_pickle('cancer_pixels_df')
cancer_pixels_df.reset_index(inplace=True)
cancer_pixels_df['index'] = cancer_pixels_df.image_name.map(
    lambda str_: str_.split('.nii.gz')[0].split('colon_')[1])

tr_fold_0_df_cancer_info = model_utils.add_cancer_pixel_info(
    df_=tr_fold_0_df.copy(), 
    cancer_pixels_df_=cancer_pixels_df)

holdout_fold_0_df_cancer_info = model_utils.add_cancer_pixel_info(
    df_=holdout_fold_0_df.copy(), 
    cancer_pixels_df_=cancer_pixels_df)

# Let's create a generator for the train and holdout set using the first fold
train_data_generator = img_generator.DataGenerator2D(
    df=tr_fold_0_df_cancer_info, x_col='x_tr_img_path', y_col='y_tr_img_path', batch_size=64,
    shuffle=True, shuffle_depths=True,
    class_sampling={'cancer_pixel': 2, 'not_cancer_pixel': 0.4}, depth_class_col='has_cancer_pixels',
    resize_dim=resize_dim, hounsfield_min=-1000., hounsfield_max=400.,
    rotate_range=30, horizontal_flip=True, vertical_flip=True, random_crop=(0.8, 0.9),
    shearing=None, gaussian_blur=None)

holdout_data_generator = img_generator.DataGenerator2D(
    df=holdout_fold_0_df, x_col='x_tr_img_path', y_col='y_tr_img_path', batch_size=32, shuffle=False,
    resize_dim=resize_dim, hounsfield_min=-1000., hounsfield_max=400.,
    rotate_range=None, horizontal_flip=False, vertical_flip=False)

num_epoch = 20
version = 1
split = '90_10_split'
lr = 1e-2
loss_used = 'binary_focal_loss'
augmentations = 'flips_rot_crop'
depth_shuffle = 'depth_shuffle'
imbalance_sampling = 'up_down_sampling'
attepmt_name_dir = f'v{version}_{num_epoch}_lr_{lr}_epochs_{split}_{loss_used}_{augmentations}_{depth_shuffle}_{imbalance_sampling}' 
attepmt_name_dir = os.path.join('training_runs', 'juan', 'binary_loss_only', attepmt_name_dir)
os.makedirs(attepmt_name_dir, exist_ok=True)

# Build model
model = custom_unet(
    input_shape=resize_dim +(1,),
    use_batch_norm=True,
    num_classes=1,
    filters=32,
    dropout=0.2,
    output_activation='sigmoid')

# TODO: save the models sumary as text
model.summary()

def scheduler(epoch, lr):
  if epoch <= 3:
    return 1e-2
  
  elif 3 < epoch <= 12:
    return 1e-3

  elif 12 < epoch <= 25:
    return 1e-4

  else:
    return 1e-5

# Set the callbacks
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1),
    tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f'./{attepmt_name_dir}' + '/model_sampling.{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir=f'./{attepmt_name_dir}/logs_new'),
]

model.compile(
    optimizer=Adam(learning_rate=lr), 
    #optimizer=SGD(lr=0.01, momentum=0.99),
    #loss='binary_crossentropy',
    loss=loss_functions.binary_focal_loss(gamma=2., alpha=0.7),
    metrics=[iou, iou_thresholded]
)

# Train the model
model.fit(train_data_generator, validation_data=holdout_data_generator,
          epochs=num_epoch, callbacks=my_callbacks)

model.save(f'./{attepmt_name_dir}/end_of_training_version')

"""It was still decreasing, so let's keep it going for 10 more epochs with lr = 1e-3"""

def scheduler(epoch, lr):
  if epoch <= 6:
    return 1e-4

  else:
    return 1e-5

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1),
    tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f'./{attepmt_name_dir}' + '/model_sampling_after_20_epochs.{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir=f'./{attepmt_name_dir}/logs_new_after_20_epochs'),
]

model.fit(train_data_generator, validation_data=holdout_data_generator,
          epochs=num_epoch, callbacks=my_callbacks)

model.save(f'./{attepmt_name_dir}/end_of_training_version_after_20_epochs')

"""Let's have it predict on the holdout data"""


iou_df, y_pred_list, y_list = model_utils.calculate_iou_holdout_set(
    holdout_df_=holdout_fold_0_df, img_dims=resize_dim,
    model_=model, pixel_threshold=0.0875, prediction_batch_size=16)

iou_df

iou_df.iou.mean()
#0.003893553145362686

"""Check mins and max of the predicted pixels to set the ranges when looking for a good threshold

"""

for y_i, y_pred_i in zip(y_list, y_pred_list):
  print(np.min(y_pred_i))

for y_i, y_pred_i in zip(y_list, y_pred_list):
  print(np.max(y_pred_i))

pd.Series(np.unique(y_pred_list[8])).plot.hist(bins=35)

"""Look for thresholds that have a better IoU


"""

iou_list = list()
min_theshold = 0
max_threshold = 1.24
n_samples = 20
true_label_threshold = 0.5
result_list = list()

for threshold in np.random.uniform(min_theshold, max_threshold, n_samples):
  for y_i, y_pred_i in zip(y_list, y_pred_list):
    img_i_iou = model_utils.calculate_iou(target=(y_i > true_label_threshold)*1,
                              prediction=(y_pred_i > threshold)*1)
    iou_list.append(img_i_iou)  

  result_list.append({'threshold': threshold, 'mean_iou':np.mean(img_i_iou)})

pd.DataFrame(result_list)

"""# Check how a given trained model predicts"""

cancer_pixels_df = pd.read_pickle('cancer_pixels_df')
cancer_pixels_df.reset_index(inplace=True)
cancer_pixels_df['index'] = cancer_pixels_df.image_name.map(lambda str_: str_.split('.nii.gz')[0].split('colon_')[1])

img_with_cancer_gen = img_generator.DataGenerator2D(df=holdout_fold_0_df_cancer_info[holdout_fold_0_df_cancer_info.cancer_pixel_area > 0].sample(20),
                                      x_col='x_tr_img_path', y_col='y_tr_img_path', batch_size=4, num_classes=None, shuffle=False, resize_dim=resize_dim)

img_without_cancer_gen = img_generator.DataGenerator2D(df=holdout_fold_0_df_cancer_info[holdout_fold_0_df_cancer_info.cancer_pixel_area==0].sample(20),
                                     x_col='x_tr_img_path', y_col='y_tr_img_path', batch_size=4, num_classes=None, shuffle=False, resize_dim=resize_dim)

#model = tf.keras.models.load_model('./5_epochs_v2', custom_objects={'iou':iou, 'iou_thresholded': iou_thresholded})

# Let's see how it predicts for images of cancer
for i, (X, y) in enumerate((img_with_cancer_gen)):
    print(f'X: {X.shape}')
    print(f'y: {y.shape}')

    y_pred = model.predict(X)
    print(y_pred.shape)
    
    for i in range(X.shape[0]):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        f.set_size_inches(20,20)

        ax1.imshow(X[i,:,:])
        ax1.set_title('Input image')
        
        ax2.imshow(y[i,:,:])
        ax2.set_title('Ground truth, target label')

        ax3.imshow(np.squeeze(y_pred[i,:,:]))
        ax3.set_title('Predicted by the model')
        
        plt.show()
        plt.close()

# Let's see how it predicts for images of cancer
for i, (X, y) in enumerate((img_without_cancer_gen)):
    print(f'X: {X.shape}')
    print(f'y: {y.shape}')

    y_pred = model.predict(X)
    print(y_pred.shape)
    
    for i in range(X.shape[0]):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        f.set_size_inches(20,20)

        ax1.imshow(X[i,:,:])
        ax1.set_title('Input image')
        
        ax2.imshow(y[i,:,:])
        ax2.set_title('Ground truth, target label')

        ax3.imshow(np.squeeze(y_pred[i,:,:]))
        ax3.set_title('Predicted by the model')
        
        plt.show()
        plt.close()



"""# Let's obtain the predictions for the test set """

path_best_performing_model = './training_runs/juan/v12_30epochs_90_10_split_jaccard_distance_loss_flips_rot_crop_depth_shuffle_up_down_sampling/end_of_training_version/'

model = tf.keras.models.load_model(
    path_best_performing_model,
    custom_objects={'iou':iou, 'iou_thresholded': iou_thresholded,
                    'binary_focal_loss_fixed': model_utils.binary_focal_loss(gamma=2., alpha=0.7)})


