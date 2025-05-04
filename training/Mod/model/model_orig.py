#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from datetime import datetime
import config

def providedModel(input_shape, learning_rate=None):
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
        
    num_labels = len(config.get_label_list())
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(pool_size=(1, 2)),
        layers.BatchNormalization(),
      
        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
      
        layers.Conv2D(128, 3, activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(256, 3, activation='relu'),
        layers.BatchNormalization(),
      
        layers.Conv2D(256, 3, activation='relu'),
        layers.BatchNormalization(),
      
        layers.GlobalMaxPooling2D(),
        layers.Dense(num_labels),
    ])
    
    # Use Adam optimizer with configurable learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    
    return model

def train_model(model, train_ds, val_ds, epochs=None):
    if epochs is None:
        epochs = config.EPOCHS
        
    history = model.fit(
        train_ds, 
        validation_data=val_ds,  
        epochs=epochs
    )
    
    return history

def evaluate_model(model, test_ds):
    # Get test data
    test_audio = []
    test_labels = []
    
    # Unbatch if necessary
    was_batched = False
    if is_batched(test_ds):
        was_batched = True
        test_ds_unbatched = test_ds.unbatch()
    else:
        test_ds_unbatched = test_ds
        
    for audio, label in test_ds_unbatched:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    # Get predictions
    model_out = model.predict(test_audio)
    y_pred = np.argmax(model_out, axis=1)
    y_true = test_labels

    # Calculate accuracy manually
    test_acc_manual = sum(y_pred == y_true) / len(y_true)
    print(f'Manual test set accuracy: {test_acc_manual:.1%}')
    
    # Use keras evaluation
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    
    # Create confusion matrix
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    
    # Calculate TPR and FPR
    label_list = config.get_label_list()
    tpr = np.zeros(len(label_list))
    fpr = np.zeros(len(label_list))
    
    for i in range(len(label_list)):
        tpr[i] = confusion_mtx[i,i] / np.sum(confusion_mtx[i,:])
        fpr[i] = (np.sum(confusion_mtx[:,i]) - confusion_mtx[i,i]) / \
          (np.sum(confusion_mtx) - np.sum(confusion_mtx[i,:]))
        print(f"True/False positive rate for '{label_list[i]:9}' = {tpr[i]:.3f} / {fpr[i]:.3f}")
    
    # Re-batch the dataset if it was batched originally
    if was_batched:
        test_ds_unbatched = test_ds_unbatched.batch(config.BATCH_SIZE)
    
    return test_loss, test_acc, confusion_mtx, tpr, fpr

def save_model(model, model_name=None):
    if model_name is None:
        model_file_name = "kws_model.h5"
    else:
        model_file_name = model_name
        
    print(f"Saving model to {model_file_name}")
    model.save(model_file_name, overwrite=True)
    
    return model_file_name

def save_model_info(model_file_name, test_acc, tpr, fpr, spectrogram_shape):
    info_file_name = model_file_name.split('.')[0] + '.txt'
    label_list = config.get_label_list()
    
    with open(info_file_name, 'w') as fpo:
        fpo.write(f"i16min            = {config.I16MIN}\n")
        fpo.write(f"i16max            = {config.I16MAX}\n")
        fpo.write(f"fsamp             = {config.FSAMP}\n")
        fpo.write(f"wave_length_ms    = {config.WAVE_LENGTH_MS}\n")
        fpo.write(f"wave_length_samps = {config.WAVE_LENGTH_SAMPS}\n")
        fpo.write(f"window_size_ms    = {config.WINDOW_SIZE_MS}\n")
        fpo.write(f"window_step_ms    = {config.WINDOW_STEP_MS}\n")
        fpo.write(f"num_filters       = {config.NUM_FILTERS}\n")
        fpo.write(f"use_microfrontend = {config.USE_MICROFRONTEND}\n")
        fpo.write(f"learning_rate     = {config.LEARNING_RATE}\n")
        fpo.write(f"label_list        = {label_list}\n")
        fpo.write(f"spectrogram_shape = {spectrogram_shape}\n")
        fpo.write(f"Test set accuracy =  {test_acc:.1%}\n")
        
        for i in range(len(label_list)):
            fpo.write(f"tpr_{label_list[i]:9} = {tpr[i]:.3f}\n")
            fpo.write(f"fpr_{label_list[i]:9} = {fpr[i]:.3f}\n")
    
    print(f"Wrote description to {info_file_name}")
    return info_file_name

def is_batched(ds):
    """Check if a dataset is already batched"""
    try:
        ds.unbatch()  # does not actually change ds
    except:
        return False
    else:
        return True