


path_best_performing_model = './training_runs/juan/v12_30epochs_90_10_split_jaccard_distance_loss_flips_rot_crop_depth_shuffle_up_down_sampling/end_of_training_version/'

model = tf.keras.models.load_model(
    path_best_performing_model,
    custom_objects={'iou':iou, 'iou_thresholded': iou_thresholded,
                    'binary_focal_loss_fixed': metric_utils.binary_focal_loss(gamma=2., alpha=0.7)})

