#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import config

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

def is_batched(ds):
    """Check if a dataset is already batched"""
    try:
        ds.unbatch()  # does not actually change ds
    except:
        return False
    else:
        return True
