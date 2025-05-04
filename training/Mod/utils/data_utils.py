#!/usr/bin/env python
# coding: utf-8

import os
import glob
import tensorflow as tf
import numpy as np
import config
from utils import audio_utils

def get_files_distribution(dataset_name=None):
    """Get training, validation and test files"""
    if dataset_name is None:
        dataset_name = config.DATASET
        
    data_dir = config.get_data_dir()
    
    # Get all wav filenames
    filenames = glob.glob(os.path.join(str(data_dir), '*', '*.wav'))
    
    # Shuffle files for randomness
    import random
    random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    
    if dataset_name == 'mini-speech':
        print('Using mini-speech')
        num_train_files = int(0.8 * num_samples) 
        num_val_files = int(0.1 * num_samples) 
        num_test_files = num_samples - num_train_files - num_val_files
        train_files = filenames[:num_train_files]
        val_files = filenames[num_train_files: num_train_files + num_val_files]
        test_files = filenames[-num_test_files:]
        
    elif dataset_name == 'full-speech-files':
        # The full speech-commands set lists which files are to be used
        # as test and validation data; train with everything else
        fname_val_files = os.path.join(data_dir, 'validation_list.txt')
        with open(fname_val_files) as fpi_val:
            val_files = fpi_val.read().splitlines()
        # validation_list.txt only lists partial paths
        val_files = [os.path.join(data_dir, fn) for fn in val_files]
        
        fname_test_files = os.path.join(data_dir, 'testing_list.txt')
        with open(fname_test_files) as fpi_tst:
            test_files = fpi_tst.read().splitlines()
        # testing_list.txt only lists partial paths
        test_files = [os.path.join(data_dir, fn).rstrip() for fn in test_files]
        
        if os.sep != '/':
            # the files validation_list.txt and testing_list.txt use '/' as path separator
            # if we're on a windows machine, replace the '/' with the correct separator
            val_files = [fn.replace('/', os.sep) for fn in val_files]
            test_files = [fn.replace('/', os.sep) for fn in test_files]
            
        # Don't train with the _background_noise_ files; exclude when directory name starts with '_'
        train_files = [f for f in filenames if f.split(os.sep)[-2][0] != '_']
        # Validation and test files are listed explicitly in *_list.txt; train with everything else
        train_files = list(set(train_files) - set(test_files) - set(val_files))
    else:
        raise ValueError("dataset must be either full-speech-files or mini-speech")
        
    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))
    
    return train_files, val_files, test_files

def limit_positive_samples(train_files):
    """Limit the number of target word samples to simulate limited data"""
    if not config.LIMIT_POSITIVE_SAMPLES:
        return train_files
        
    commands = config.COMMANDS
    max_wavs_0 = config.MAX_WAVS_0
    max_wavs_1 = config.MAX_WAVS_1
    
    # Filter files based on command limits
    filtered_files = []
    num_files_cmd0 = 0
    num_files_cmd1 = 0
    
    for f in train_files:
        cmd = f.split(os.sep)[-2]
        
        if cmd == commands[0]:
            if num_files_cmd0 < max_wavs_0:
                filtered_files.append(f)
                num_files_cmd0 += 1
        elif cmd == commands[1]:
            if num_files_cmd1 < max_wavs_1:
                filtered_files.append(f)
                num_files_cmd1 += 1
        else:
            # Keep non-command files
            filtered_files.append(f)
    
    print(f"Limited samples: {commands[0]}={num_files_cmd0}, {commands[1]}={num_files_cmd1}")
    return filtered_files

def limit_silence_unkown_samples(train_files):
    """Limit the number of '_silence_' and '_unknown_' samples"""
    if not config.LIMIT_OTHER_SAMPLES:
        return train_files

    max_silence = config.MAX_SILENCE_SAMPLES
    max_unknown = config.MAX_UNKNOWN_SAMPLES
    commands = config.COMMANDS

    silence_count = 0
    unknown_count = 0
    cmd_counts = {cmd: 0 for cmd in commands}
    
    filtered_files = []
    
    for f in train_files:
        cmd = f.split(os.sep)[-2]

        if cmd == config.SILENCE_STR:
            if silence_count < max_silence:
                filtered_files.append(f)
                silence_count += 1
        elif cmd not in commands:
            if unknown_count < max_unknown:
                filtered_files.append(f)
                unknown_count += 1
        else:
            # Always include target command files
            filtered_files.append(f)
            cmd_counts[cmd] += 1

    print(f"Limited samples: _silence_={silence_count}, _unknown_={unknown_count}, "
          f"{commands[0]}={cmd_counts[commands[0]]}, {commands[1]}={cmd_counts[commands[1]]}")
    
    return filtered_files

def preprocess_dataset(files, num_silent=None, noisy_reps_of_known=None):
    """
    Preprocess dataset files into spectrograms
    
    Parameters:
    files -- list of files
    num_silent -- number of silent samples to create
    noisy_reps_of_known -- list of rms noise levels to add to target words
    """
    if num_silent is None:
        num_silent = int(0.2 * len(files)) + 1
        
    print(f"Processing {len(files)} files")
    
    # Create dataset from files
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    waveform_ds = files_ds.map(audio_utils.get_waveform_and_label, 
                              num_parallel_calls=config.AUTOTUNE)
    
    # Add noisy copies of target words if specified
    if noisy_reps_of_known is not None:
        # Create a tmp dataset with only the target words
        ds_only_cmds = waveform_ds.filter(
            lambda w, l: tf.reduce_any(l == config.COMMANDS))
        
        for noise_level in noisy_reps_of_known:
            waveform_ds = waveform_ds.concatenate(
                audio_utils.copy_with_noise(ds_only_cmds, rms_level=noise_level))
    
    # Add silence samples
    if num_silent > 0:
        silent_wave_ds = audio_utils.create_silence_dataset(
            num_silent, 
            rms_noise_range=[0.01, 0.2], 
            silent_label=config.SILENCE_STR)
        waveform_ds = waveform_ds.concatenate(silent_wave_ds)
    
    print(f"Added {num_silent} silent wavs and noisy samples")
    
    # Convert waveforms to spectrograms
    output_ds = audio_utils.wavds2specds(waveform_ds)
    return output_ds

def prepare_datasets(train_files, val_files, test_files):
    """
    Prepare training, validation, and test datasets
    
    Parameters:
    train_files -- list of training files
    val_files -- list of validation files
    test_files -- list of test files
    
    Returns:
    train_ds, val_ds, test_ds, input_shape
    """
    # Apply sample limits if configured
    if config.LIMIT_POSITIVE_SAMPLES:
        train_files = limit_positive_samples(train_files)


    if config.LIMIT_OTHER_SAMPLES:
        train_files = limit_silence_unkown_samples(train_files)
    
    print(f"Preparing datasets from {len(train_files)}/{len(val_files)}/{len(test_files)} files")
    
    # Process all datasets
    with tf.device('/CPU:0'):  # needed on M1 mac
        train_ds = preprocess_dataset(
            train_files, 
            noisy_reps_of_known=[0.05, 0.1, 0.15, 0.2, 0.25, 0.1, 0.1, 0.1])
    
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)
    
    # Shuffle datasets
    train_ds = train_ds.shuffle(int(len(train_files) * 1.2))
    val_ds = val_ds.shuffle(int(len(val_files) * 1.2))
    test_ds = test_ds.shuffle(int(len(test_files) * 1.2))
    
    # Batch and cache datasets
    train_ds = train_ds.batch(config.BATCH_SIZE)
    val_ds = val_ds.batch(config.BATCH_SIZE)
    test_ds = test_ds.batch(config.BATCH_SIZE)
    
    train_ds = train_ds.cache().prefetch(config.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(config.AUTOTUNE)
    
    # Get input shape from the first batch
    for spectrogram, _ in train_ds.take(1):
        input_shape = spectrogram.shape[1:]
    
    print('Input shape:', input_shape)
    
    return train_ds, val_ds, test_ds, input_shape

def is_batched(ds):
    """Check if a dataset is already batched"""
    try:
        ds.unbatch()  # does not actually change ds
    except:
        return False
    else:
        return True

def count_labels(dataset):
    """Count occurrences of each label in a dataset"""
    counts = {}
    for sample in dataset:
        lbl = sample[1]
        if lbl.dtype == tf.string:
            label = lbl.numpy().decode('utf-8')
        else:
            label = lbl.numpy()
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return counts