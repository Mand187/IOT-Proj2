#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
import os

# Ensure the output directory exists
os.makedirs('output/graphs', exist_ok=True)

def plot_confusion_matrix(confusion_mtx, label_list=None):
    """
    Plot confusion matrix
    
    Parameters:
    confusion_mtx -- confusion matrix tensor
    label_list -- list of labels (default: use config label list)
    """
    if label_list is None:
        label_list = config.get_label_list()
        
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_mtx, xticklabels=label_list, yticklabels=label_list, 
                annot=True, fmt='g')
    plt.gca().invert_yaxis()  # flip so origin is at bottom left
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.savefig('output/graphs/confusion_matrix.png')
    plt.show()

def plot_training_history(history):
    """
    Plot training and validation loss/accuracy
    
    Parameters:
    history -- training history object
    """
    metrics = history.history
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogy(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['training', 'validation'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/graphs/training_curves.png')
    plt.show()

def plot_example_spectrograms(dataset, num_examples=3):
    """
    Plot example spectrograms from dataset
    
    Parameters:
    dataset -- dataset containing spectrograms
    num_examples -- number of examples to plot
    """
    label_list = config.get_label_list()
    plt.figure(figsize=(10, 10))
    
    for i, (spectrogram, label_id) in enumerate(dataset.unbatch().take(num_examples)):
        if i >= num_examples:
            break
            
        # Get the spectrogram and remove extra dimensions
        spec = spectrogram.numpy()
        spec = np.squeeze(spec)  # Remove channel dimension
        
        plt.subplot(num_examples, 1, i+1)
        plt.imshow(spec, aspect='auto', origin='lower')
        plt.title(f"Label: {label_list[int(label_id)]}")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
    
    plt.savefig('output/graphs/example_spectrograms.png')
    plt.show()

def visualize_model_performance(confusion_mtx, tpr, fpr):
    """
    Create visual report of model performance
    
    Parameters:
    confusion_mtx -- confusion matrix
    tpr -- true positive rates
    fpr -- false positive rates
    """
    label_list = config.get_label_list()
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion_mtx, label_list)
    
    # Plot TPR and FPR as bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(label_list))
    width = 0.35
    
    plt.bar(x - width/2, tpr, width, label='True Positive Rate')
    plt.bar(x + width/2, fpr, width, label='False Positive Rate')
    
    plt.xlabel('Command')
    plt.ylabel('Rate')
    plt.title('True Positive and False Positive Rates')
    plt.xticks(x, label_list, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('output/graphs/tpr_fpr.png')
    plt.show()