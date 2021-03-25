import yaml

config_dict = {
    'training_params': {
        'learning_rate': 1e-2,
        'batch_size_train': 64,
        'batch_size_val': 64,
        'num_epoch': 30,
        'loss_function_name': 'binary_focal_loss',
        'loss_function_params': {'gamma': 2., 'alpha': 0.7},
        'num_filters_first_level_': 32,
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
        }
    },

    'model_params': {
        'output_threshold': 0.0875,
        'best_model_path': '',
        'predict_params': {
            'prediction_batch_size': 32,
            'output_dir': 'test_pred',
            'test_dims': (512, 512)
        }
    }
}

with open('config.yml', 'w') as outfile:
    yaml.dump(config_dict, outfile, default_flow_style=False)
