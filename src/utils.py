import scipy.misc
import numpy as np

def merge_image(images, size=(10, 10)):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h*size[0], w*size[1], 3))
    for idx in range(min(size[0] * size[1], len(images))):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:(j+1)*h, i*w:(i+1)*w, :] = images[idx, :, :, :]
    return img

def save_images(path, images, size=(10, 10)):
    return scipy.misc.imsave(path, merge_image(images, size=size))
