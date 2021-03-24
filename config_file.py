training_params = {
    'learning_rate': 1e-2,
    'batch_size_train': 64,
    'batch_size_val': 64,
    'num_epoch': 30,
    'loss_function_name': 'binary_focal_loss',
    'model_object_storing_dir': 'model_training_data',
    'preprocess_object_storing_dir': 'preprocessed_dataframes',
    'folds': 10,
    'seed': 123,

    'preprocesing_params': {
        'resize_dim': (128, 128),
        'hounsfield_min': -1000.,
        'hounsfield_max': 400.
    },

    'augmentation_params': {
        'rotate_range': 30,
        'horizontal_flip': True,
        'vertical_flip': True,
        'random_crop': (0.8, 0.9),
        'shearing': None,
        'gaussian_blur': None
    },

    'sampling_params': {
        'class_sampling': {'cancer_pixel': 2, 'not_cancer_pixel': 0.4},
        'depth_class_col': 'has_cancer_pixels'
    },

}
