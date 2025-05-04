#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
import config

def decode_audio(audio_binary):
    """Decode audio binary data to waveform"""
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    """Extract label from file path"""
    parts = tf.strings.split(file_path, os.path.sep)
    label_list = config.get_label_list()
    in_set = tf.reduce_any(parts[-2] == label_list)
    label = tf.cond(in_set, lambda: parts[-2], lambda: tf.constant(config.UNKNOWN_STR))
    return label

def get_waveform_and_label(file_path):
    """Get waveform data and corresponding label from a file path"""
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram(waveform):
    """Convert waveform to spectrogram using microfrontend"""
    # Concatenate audio with padding so that all audio clips will be of the same length
    zero_padding = tf.zeros([config.WAVE_LENGTH_SAMPS] - tf.shape(waveform), dtype=tf.int16)
    waveform = tf.cast(0.5*waveform*(config.I16MAX-config.I16MIN), tf.int16)  # scale float [-1,+1]=>INT16
    equal_length = tf.concat([waveform, zero_padding], 0)
    
    # Generate the spectrogram using microfrontend
    spectrogram = frontend_op.audio_microfrontend(
        equal_length, 
        sample_rate=config.FSAMP, 
        num_channels=config.NUM_FILTERS,
        window_size=config.WINDOW_SIZE_MS, 
        window_step=config.WINDOW_STEP_MS
    )
    return spectrogram

def pad_waveform(waveform, label):
    """Pad waveform to ensure consistent length"""
    zero_padding = tf.zeros([config.WAVE_LENGTH_SAMPS] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)        
    return waveform, label

def copy_with_noise(ds_input, rms_level=0.25):
    """Create a copy of the dataset with added noise"""
    rng = tf.random.Generator.from_seed(1234)
    wave_shape = tf.constant((config.WAVE_LENGTH_SAMPS,))
    
    def add_noise(waveform, label):
        noise = rms_level*rng.normal(shape=wave_shape)
        zero_padding = tf.zeros([config.WAVE_LENGTH_SAMPS] - tf.shape(waveform), dtype=tf.float32)
        waveform = tf.concat([waveform, zero_padding], 0)    
        noisy_wave = waveform + noise
        return noisy_wave, label

    return ds_input.map(add_noise)

def create_silence_dataset(num_waves, rms_noise_range=[0.01, 0.2], silent_label=config.SILENCE_STR):
    """Create silent waveforms with a small amount of noise"""
    rng = np.random.default_rng()
    rms_noise_levels = rng.uniform(low=rms_noise_range[0], high=rms_noise_range[1], size=num_waves)
    rand_waves = np.zeros((num_waves, config.WAVE_LENGTH_SAMPS), dtype=np.float32)
    
    for i in range(num_waves):
        rand_waves[i,:] = rms_noise_levels[i] * rng.standard_normal(config.WAVE_LENGTH_SAMPS)
    
    labels = [silent_label] * num_waves
    return tf.data.Dataset.from_tensor_slices((rand_waves, labels))

def wavds2specds(waveform_ds, verbose=True):
    """Convert waveform dataset to spectrogram dataset"""
    wav, label = next(waveform_ds.as_numpy_iterator())
    one_spec = get_spectrogram(wav)
    one_spec = tf.expand_dims(one_spec, axis=0)  # add a 'batch' dimension at the front
    one_spec = tf.expand_dims(one_spec, axis=-1) # add a singleton 'channel' dimension at the back    

    # Count waveforms for memory allocation
    num_waves = 0
    for wav, label in waveform_ds:
        num_waves += 1
    
    print(f"About to create spectrograms from {num_waves} waves")
    spec_shape = (num_waves,) + one_spec.shape[1:] 
    spec_grams = np.nan * np.zeros(spec_shape)  # allocate memory
    labels = np.nan * np.zeros(num_waves)
    
    idx = 0
    label_list = config.get_label_list()
    
    for wav, label in waveform_ds:    
        if verbose and idx % 250 == 0:
            print(f"\r {idx} wavs processed", end='')
        
        spectrogram = get_spectrogram(wav)
        # Add dimensions for TF conv layer expectations (batch_size, height, width, channels)
        spectrogram = tf.expand_dims(spectrogram, axis=0)
        spectrogram = tf.expand_dims(spectrogram, axis=-1)
        
        spec_grams[idx, ...] = spectrogram
        new_label = label.numpy().decode('utf8')
        new_label_id = np.argmax(new_label == np.array(label_list))    
        labels[idx] = new_label_id
        idx += 1
    
    labels = np.array(labels, dtype=int)
    output_ds = tf.data.Dataset.from_tensor_slices((spec_grams, labels))  
    return output_ds

import os  # Add this at the top with other imports