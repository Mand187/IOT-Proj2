import os
import platform
import numpy as np
import tensorflow as tf

# Set seed for experiment reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Audio parameters
I16MIN = -2**15
I16MAX = 2**15-1
FSAMP = 16000  # sampling rate
WAVE_LENGTH_MS = 1000  # 1000 => 1 sec of audio
WAVE_LENGTH_SAMPS = int(WAVE_LENGTH_MS*FSAMP/1000)

# Feature extraction parameters
WINDOW_SIZE_MS = 64
WINDOW_STEP_MS = 48
NUM_FILTERS = 32
USE_MICROFRONTEND = True  # recommended, but you can use another feature extractor if you like

# Dataset configuration
DATASET = 'mini-speech'  # 'mini-speech' or 'full-speech-files'
COMMANDS = ['left', 'right']  # Change this line for your custom keywords
SILENCE_STR = "_silence"  # label for <no speech detected>
UNKNOWN_STR = "_unknown"  # label for <speech detected but not one of the target words>

# Limit the instances of each command in the training set to simulate limited data
LIMIT_POSITIVE_SAMPLES = True
MAX_WAVS_0 = 50  # use no more than ~ samples of commands[0]
MAX_WAVS_1 = 250  # use no more than ~ samples of commands[1]

LIMIT_OTHER_SAMPLES = True
MAX_SILENCE_SAMPLES = 200
MAX_UNKNOWN_SAMPLES = 500

# Training parameters
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001  # Default learning rate, can be overridden with command line args

# System configuration
APPLE_SILICON = platform.processor() == 'arm'
AUTOTUNE = tf.data.AUTOTUNE

def get_label_list():
    """Generate the complete label list including silence and unknown"""
    label_list = COMMANDS.copy()
    label_list.insert(0, SILENCE_STR)
    label_list.insert(1, UNKNOWN_STR)
    return label_list

def get_data_dir():
    """Returns the path to the data directory based on the dataset configuration"""
    home_dir = os.getcwd()  # Define this at the top for both branches

    if DATASET == 'mini-speech':
        data_dir = os.path.join(home_dir, 'data', 'mini_speech_commands')
        if not os.path.exists(data_dir):
            tf.keras.utils.get_file(
                'mini_speech_commands.zip',
                origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                extract=True, cache_dir='.', cache_subdir='data'
            )

    elif DATASET == 'full-speech-files':
        data_dir = os.path.join(home_dir, 'data', 'speech_commands_v0.02')
        if not os.path.exists(data_dir):
            tf.keras.utils.get_file(
                'speech_commands_v0.02.tar.gz',
                origin="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                extract=True, cache_dir='.', cache_subdir='data'
            )
    else:
        raise RuntimeError('dataset should either be "mini-speech" or "full-speech-files"')

    return data_dir


def print_configuration():
    """Print the current configuration settings"""
    print(f"FFT window length = {int(WINDOW_SIZE_MS * FSAMP / 1000)}")
    print(f"Learning rate: {LEARNING_RATE}")
    label_list = get_label_list()
    print(f"Commands: {COMMANDS}")
    print(f"Label list: {label_list}")