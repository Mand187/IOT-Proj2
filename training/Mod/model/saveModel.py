#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import config
import tensorflow as tf

# Ensure the output directory exists
OUTPUT_DIR_TEXT = "output/text"
OUTPUT_DIR_MODEL = "output/model"

os.makedirs(OUTPUT_DIR_MODEL, exist_ok=True)

def save_model(model, model_name=None):
    if model_name is None:
        model_file_name = "kws_model.h5"
    else:
        model_file_name = model_name + ".h5"
    
    model_file_path = os.path.join(OUTPUT_DIR_MODEL, model_file_name)
    print(f"Saving model to {model_file_path}")
    model.save(model_file_path, overwrite=True)
    
    return model_file_path

def convert_and_save_tflite_model(model_path, val_ds, tflite_file_name, num_calibration_steps=10):
    # Load the Keras model from .h5 file
    model = tf.keras.models.load_model(model_path)

    # Set up TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Disable experimental lowering of tensor list ops
    converter._experimental_lower_tensor_list_ops = False

    # Enable Select TensorFlow ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    # Prepare representative dataset generator
    ds_iter = val_ds.unbatch().batch(1).as_numpy_iterator()

    def representative_dataset_gen():
        for _ in range(num_calibration_steps):
            next_input = next(ds_iter)[0]
            next_input = next_input.astype(np.float32)
            yield [next_input]

    converter.representative_dataset = representative_dataset_gen
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert the model to TensorFlow Lite format
    tflite_quant_model = converter.convert()
    tflite_file_name = tflite_file_name + ".tflite"

    # Save the TensorFlow Lite model to a .tflite file in the same output folder
    tflite_file_path = os.path.join(OUTPUT_DIR_MODEL, tflite_file_name)
    with open(tflite_file_path, "wb") as fpo:
        num_bytes_written = fpo.write(tflite_quant_model)
    print(f"Wrote {num_bytes_written} / {len(tflite_quant_model)} bytes to tflite file at {tflite_file_path}")

    return tflite_file_path

def save_model_info(model_file_name, test_acc, tpr, fpr, spectrogram_shape):
    info_file_name = os.path.splitext(model_file_name)[0] + '.txt'
    info_file_path = os.path.join(OUTPUT_DIR_TEXT, os.path.basename(info_file_name))
    label_list = config.get_label_list()
    
    with open(info_file_path, 'w') as fpo:
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
    
    print(f"Wrote description to {info_file_path}")
    return info_file_path
