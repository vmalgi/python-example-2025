#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import xgboost as xgb
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files from both datasets
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    # Initialize arrays for combined data, 7 features based on the current extract_features function
    # Keep 7 features based on the current extract_features function
    features = np.zeros((num_records, 7), dtype=np.float64) 
    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record)
        labels[i] = load_label(record)

    # Calculate scale_pos_weight for class imbalance
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    if verbose:
        print(f'Training data label distribution: Positive={num_positive}, Negative={num_negative}')
    if num_positive > 0:
        scale_pos_weight_value = num_negative / num_positive
        if verbose:
            print(f'Calculated scale_pos_weight: {scale_pos_weight_value:.2f}')
    else:
        scale_pos_weight_value = 1 # Default value if no positive samples
        if verbose:
            print('Warning: No positive samples found in training data. Using default scale_pos_weight=1')

    # Train the models.
    if verbose:
        print('Training the model on the data...')

    # Define the parameters for the XGBoost model
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 56,
        'scale_pos_weight': scale_pos_weight_value # Add calculated weight here
    }

    # Fit the model
    model = xgb.XGBClassifier(**params).fit(features, labels)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    # Load the model.
    model_filename = os.path.join(model_folder, 'model.sav')
    if os.path.exists(model_filename):
        if verbose >= 1:
            print('Loading the Challenge model...')
        # Load the dictionary and extract the model
        loaded_data = joblib.load(model_filename)
        return loaded_data['model']
    else:
        if verbose >= 1:
            print('No saved model found. Returning None.')
        return None # Indicate model needs training or wasn't saved

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Extract features.
    features = extract_features(record)
    features = features.reshape(1, -1) # Reshape for single sample prediction

    # Make predictions.
    if model is None:
        raise ValueError("Model is None, cannot make predictions.")

    # Get probabilities for the positive class (class 1)
    probabilities = model.predict_proba(features)
    probability_positive = probabilities[0, 1] # Assuming positive class is index 1

    # Choose the class based on the probability and threshold.
    threshold = 0.60
    label = 1 if probability_positive >= threshold else 0

    if verbose >= 2:
        print(f"Record: {get_record_name(load_header(record))}, Probability: {probability_positive:.4f}, Predicted Label: {label}")

    return label, probability_positive

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # Calculate basic signal statistics
    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0
        
    # Calculate heart rate variability
    hrv = 0.0
    if num_finite_samples > 1:
        # Find peaks in the signal (R peaks)
        peaks = np.where((signal[1:-1] > signal[:-2]) & 
                        (signal[1:-1] > signal[2:]))[0] + 1
        if len(peaks) > 1:
            # Calculate intervals between peaks
            intervals = np.diff(peaks)
            # Calculate HRV as the standard deviation of intervals
            hrv = np.std(intervals) if len(intervals) > 1 else 0.0

    # Combine all features including HRV
    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std, hrv]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)