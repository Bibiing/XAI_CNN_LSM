import numpy as np
import random

def shuffle_image_label(images, labels):
    """
    data shuffle
    """
    randnum = random.randint(0, len(images))
    random.seed(randnum)
    random.shuffle(images)
    random.seed(randnum)
    random.shuffle(labels)
    return images, labels

def get_CNN_data(feature_block, label_raster, window_size):
    """
    Creating a CNN dataset
    """
    n = window_size // 2
    train_imgs, train_labels = [], []
    val_imgs, val_labels = [], []

    # train data (label 0 dan 2)
    train_longsor_y, train_longsor_x = np.where(label_raster == 0)
    for y, x in zip(train_longsor_y, train_longsor_x):
        patch = feature_block[:, y-n:y+n+1, x-n:x+n+1]
        train_imgs.append(patch)
        train_labels.append(0) # Label: landslide

    train_aman_y, train_aman_x = np.where(label_raster == 2)
    for y, x in zip(train_aman_y, train_aman_x):
        patch = feature_block[:, y-n:y+n+1, x-n:x+n+1]
        train_imgs.append(patch)
        train_labels.append(1) # Label: no landslides

    # Val data (label 1 dan 3)
    val_longsor_y, val_longsor_x = np.where(label_raster == 1)
    for y, x in zip(val_longsor_y, val_longsor_x):
        patch = feature_block[:, y-n:y+n+1, x-n:x+n+1]
        val_imgs.append(patch)
        val_labels.append(0) # Label: landslide
        
    val_aman_y, val_aman_x = np.where(label_raster == 3)
    for y, x in zip(val_aman_y, val_aman_x):
        patch = feature_block[:, y-n:y+n+1, x-n:x+n+1]
        val_imgs.append(patch)
        val_labels.append(1) # Label: no landslides
        
    # Acak dan kembalikan sebagai NumPy array
    train_imgs, train_labels = shuffle_image_label(train_imgs, train_labels)
    val_imgs, val_labels = shuffle_image_label(val_imgs, val_labels)

    return (np.array(train_imgs), np.array(train_labels).reshape(-1, 1), np.array(val_imgs), np.array(val_labels).reshape(-1, 1))

