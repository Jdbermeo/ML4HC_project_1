import os

import tensorflow as tf

from model import training_utils, model_utils


def train(data_path_source_dir_: str, training_params: dict):
    model_object_storing_dir = training_params['model_object_storing_dir']
    os.makedirs(model_object_storing_dir, exist_ok=True)

    preprocesing_params_ = training_params['preprocesing_params']

    # Get data generators for train and holdout
    train_data_generator, holdout_data_generator = training_utils.repare_train_holdout_generators(
        data_path_source_dir_=data_path_source_dir_, training_params=training_params)

    # Build model
    model = model_utils.create_model(
        resize_dim_=preprocesing_params_['resize_dim'],
        lr_=training_params['learning_rate'],
        loss_function_name_=training_params['loss_function_name'],
        object_storing_dir_=model_object_storing_dir,
        num_filters_first_level_=32, gamma=2., alpha=0.7
    )

    # Set the callbacks
    my_callbacks = [
        tf.keras.callbacks.LearningRateScheduler(training_utils.scheduler, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'./{model_object_storing_dir}' + '/model_sampling.{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=f'./{model_object_storing_dir}/logs'),
    ]

    # Train the model
    model.fit(train_data_generator, validation_data=holdout_data_generator,
               epochs=training_params['num_epoch'], callbacks=my_callbacks)

    model.save(f'./{model_object_storing_dir}/end_of_training_version')
