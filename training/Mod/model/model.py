#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers

import numpy as np
import tensorflow as tf
import pandas as pd

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

def modelL1Reg(input_shape, learning_rate=None):
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
        
    num_labels = len(config.get_label_list())
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        layers.MaxPooling2D(pool_size=(1, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        layers.BatchNormalization(),

        layers.Conv2D(256, 3, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
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

def modelL2Reg(input_shape, learning_rate=None):
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
        
    num_labels = len(config.get_label_list())
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling2D(pool_size=(1, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),

        layers.Conv2D(256, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
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

def projectModel(input_shape, learning_rate=None, l2=0.01):
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE

    num_labels = len(config.get_label_list())

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, kernel_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(name='pool1', padding='same'),

        layers.Conv2D(64, 3, kernel_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(name='pool2', padding='same'),

        layers.Conv2D(128, 3, kernel_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(name='pool3', padding='same'),

        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(128, kernel_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(num_labels),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    return model


def measure_sparsity(model):
    layer_sparsity = []
    
    # Loop through the layers of the model
    for layer in model.layers:
        # Only check Conv2D and Dense layers
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights = layer.get_weights()[0]  # Get the weights of the layer (weights[0] is the actual weight matrix)
            
            # Get the maximum absolute value in the layer
            max_abs_value = np.max(np.abs(weights))
            
            # Define "close to zero" as values less than 0.01 * max absolute value
            threshold = 0.01 * max_abs_value
            
            # Count the number of values that are close to zero
            close_to_zero_count = np.sum(np.abs(weights) < threshold)
            
            # Total number of weights in the layer
            total_weights = weights.size
            
            # Sparsity is the ratio of "close to zero" values to total weights
            sparsity = close_to_zero_count / total_weights
            
            # Store the sparsity for this layer
            layer_sparsity.append({
                'Layer': layer.name,
                'Sparsity': sparsity
            })
    
    return layer_sparsity

def get_sparsity_table(model_list):
    # List to store sparsity data for each model
    all_sparsity = []
    
    for model_name, model in model_list.items():
        sparsity = measure_sparsity(model)
        for layer in sparsity:
            layer['Model'] = model_name
        all_sparsity.extend(sparsity)
    
    # Convert the results to a pandas DataFrame for easier viewing
    df = pd.DataFrame(all_sparsity)
    
    return df