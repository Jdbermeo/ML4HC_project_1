# Environment

To replicate the environment please use `environment_droplet.yml`

# Training

To run the training script, run `main.py --mode train --config_file_path config.yml` 

The `config.yml` has all input variables for training like the loss function to be used, number of epochs, augmenation operations to use, sampling, etc, with the exception of the learning rate's schedule. To modify the learning rate's schedule please modify it in `model/training_utils.py` in the `scheduler()` function.

**The relevant sections of `config.yml` for training are:**

 - data_params: Dictionary with parameters related to the data. Currently it is only the path to the directory where the data is stored with the imageTs/Tr and lablesTr structure.
 
 
 - training_params: Dictionary with all input values needed for the training such as:
     - batch_size, number of folds/splits to use, loss function to use, number epochs, number of filters at the first level of Unet, directory to store model objects
     - preprocesing_params: Resizing for the images at training and capping values before normalizing the pixels HU value.
     - augmentation_params: Which augmentation operations to use and their respective parameters (To check what the augmentation operations do please check `notebooks/Test augmentation transformations.ipynb`)
     - sampling_params: Weights to upsample and/or downsample slices that contain and do not contain cancer, respectively.


After training, you can use `notebooks/Tune threshold.ipynb` to look for an appropriate threshold for any of the checkpoints of the model based on the IoU on the holdout set.

Likewise, you can also use `notebooks/Check predictions on individual slices.ipynb` To check how a given model checkpoint predicts on inivdual slices that do or do not have cancer with both the train and holdout set 

# Predict

**The relevant sections of `config.yml` for training are:**

 - data_params: Dictionary with parameters related to the data. Currently it is only the path to the directory where the data is stored with the imageTs/Tr and lablesTr structure.
 
 - model_params: Parameters of the model during its use for prediction and of its best perfroming model
     - Where the model to be used for prediction is located and the threshold chosen for it
     - predict_params: Dictionary with params such as name of the dir to store the predicted images in .npz format, batch size to use in prediction and the dimension of the test images (to resize predicted outputs to this size) 
     
Predicted results are currently stored at `test_pred`

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
     
We also have a and end to end example (preprocessing to obtaining predictions) notebook (`end to end example.ipynb`) in which we use the functions in the different scripts. This notebook was used initally for experimenting multiple hyper-parameters but now serves as reference of the logic that the scripts follow.

# Possible next steps

 - Switch from a single channel to a multi-channel (or depth) approach in which we would have the 2D Unet predict using multiple slices at the same time instead of one. This seems rather promising as cancer labels occurred mostly in one consecutive set of slices.
 - Exacly for the same reason, use a 3D Unet. As all the images have different depths, and many of them are not powers of 2, we would need define mutiple instances of each image in groups of slices of its depth that are powers of 2 
