# Environment

To replicate the environment please use `environment.yml`.

# Training

To run the training script run: 


```python
python main.py --mode train --config_file_path config.yml 
```

The `config.yml` has all input variables for training like the loss function to be used, number of epochs, augmenation operations to use, sampling, etc., with the exception of the learning rate's schedule. To modify the learning rate's schedule please modify it in `model/training_utils.py` in the `scheduler()` function.

**The relevant sections of `config.yml` for training are:**

 - data_params: Dictionary with parameters related to the data. Currently it is only the path to the directory where the data is stored with the imageTs/Tr and lablesTr subdir structure.
 
 
 - training_params: Dictionary with all input values needed for the training such as:
     - batch_size, number of folds/splits, loss function, number epochs, number of filters at the first level of Unet, directory to store model objects, etc.
     - preprocesing_params: New dimensions for the images at training. Capping pixel HU values before normalizing.
     - augmentation_params: Which augmentation operations to use and their respective parameters (to check what the augmentation operations do please check `notebooks/Test augmentation transformations.ipynb`).
     - sampling_params: Weights to upsample and/or downsample slices that contain and do not contain cancer, respectively.

After training, you can use `notebooks/Tune threshold.ipynb` to look for an appropriate threshold for any of the checkpoints of the model based on the IoU on the holdout set.

Likewise, you can also use `notebooks/Check predictions on individual slices.ipynb` To check how a given model checkpoint predicts on inivdual slices that do or do not have cancer with both the train and holdout set.

# Predict

To run the script that generates the predicted mask for the test set run: 


```python
python main.py --mode test --config_file_path config.yml 
```

**The relevant sections of `config.yml` for training are:**

 - data_params: Dictionary with parameters related to the data. Currently it is only the path to the directory where the data is stored with the imageTs/Tr and lablesTr subdir structure.
 
 - model_params: Parameters of the model during its use for prediction and the path to the `.h5` file of the best perfroming model
     - Where the model to be used for prediction is located and the threshold chosen for it.
     - predict_params: Dictionary with parameters such as the name of the diretory to store the predicted images in `.npz` format, mini-batch size to use in prediction, and the 2D dimension of the test images (so that we can resize predicted images to this size). 
     
Predicted results are currently stored at a directory named `test_pred`.

# Notebooks

We wrote some notebooks to do the following:

 - Analyze the data given (images and labels):
     - EDA: `notebooks/EDA.ipynb`
     - Verify the imablance in the classes: `notebooks/Check imbalance of the data.ipynb`
     - Check label masks: `notebooks/check_all_masks.ipynb`
     
     
     
 - Check the behavior of the pipeline in terms of
     - Augmentation transformations: `notebooks/Test augmentation transformations.ipynb`
     - Functions used to measure the IoU: `notebooks/Test IoU script.ipynb`
 - Behaviour of predictions
     - Tune the threshold: `notebooks/Tune threshold.ipynb`
     - Check predictions on individual slices: `notebooks/Check predictions on individual slices.ipynb`
     
We also have an example notebook where we do all the steps in the scripts (from preprocessing to obtaining predictions) called `end to end example.ipynb` in which we use the functions in the different scripts. This notebook was used initally for experimenting multiple hyper-parameters but now serves as reference of the logic that the scripts follow.

# Scripts

The scripts used to train a Unet and generate the predictons are the following:

 - preprocessing:
     - `get_ct_scan_information.py`: We use functions ins this script to generate a dataframe with the paths of the images to use and information about them like their depth, and which of their slices in the z dimension contain cancer pixels or not. These dataframes have one row per slice in the z-dim per image. The datframes created with the functions in it serves as the input for the script that defines the class DataGenerator2D.
  
  - model:
      - `img_generator.py`: Script where we extend the `tf.keras.utils.Sequence` to read `.nii.gz` files, operate or not augmentation transformations on the fly, up sample or down sample slices with cancer or not, and generate minibatches of different 2D single channel slices for the 3D CT scans to feed a `keras.model` during the `fit()` operation.
      - `loss_functions.py`: Definition of some custome loss functions we can use for the Unet.
      - `training_utils.py`: Script that has helper functions to execute the training of the Unet like the one used to create the folds and prepare the generators for each fold that will feed the model, a function to choose a loss function to use, or the leraning rate's schedule. This script uses `get_ct_scan_information.py`, `get_ct_scan_information.py`, and `loss_functions.py`.
      - `model_utils.py`: Helper functions to create the Unet and compile it with a given loss function.
      - `metrics_utils.py`: Script with functions used to evaluate the IoU for the images of the holdout set.
      - `train.py`: Function that concatenates all previous scripts to create a Unet with the parameters specified in `config.yml` and execute the training and store it's progress, results and performance on the holdout set.
      - `predict.py`: Script that loads a model spcified in `config.yml` and generates predictions for the images of the test set and stores the predictions in `.npz` format.
      
  - main:
      - Run in either `train` or `predict` mode which calls on the corresponding functions and scripts of the same name.

# Possible next steps

 - Switch from a single channel to a multi-channel (or depth) approach in which we would have the 2D Unet predict using multiple slices at the same time instead of one. This seems rather promising as cancer labels occurred mostly in one consecutive set of slices.
 - Exacly for the same reason, use a 3D Unet. As all the images have different depths, and many of them are not powers of 2, we would need define mutiple instances of each image in groups of slices of its depth that are powers of 2.
