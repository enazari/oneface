from utils.Image import img_resize_pil

import numpy as np

from mtcnn.mtcnn import MTCNN

_mtcnn = MTCNN()


def face_detection(img_):
    output = _mtcnn.detect_faces(img_)
    (x, y, width, height) = output[0]['box']

    # a bug in MTCNN: https://github.com/ipazc/mtcnn/issues/11
    if x < 0: x = 0
    if y < 0: y = 0

    return img_[y:y + height,
           x:x + width, :]


def make_image_fit_to_mtcnn_one_time(img, destination_image_dim):
    '''
    makes the input image fit completely for mtcnn, so that if mtcnn is called on output image,
    the same exact image will be returned.
    for example if the input image is 160 by 160, the output of mtcnn is usually less than it according to
    the humans face position. this method makes the image to be output of mtcnn of exact size of 160 by 160
    '''

    img = face_detection(img)
    img = img_resize_pil(img, destination_image_dim)
    img = np.asarray(img, 'uint8')

    return img


def make_image_fit_to_mtcnn(img, destination_image_dim):
    '''
    makes the input image fit completely for mtcnn, so that if mtcnn is called on output image,
    the same exact image will be returned.
    for example if the input image is 160 by 160, the output of mtcnn is usually less than it according to
    the humans face position. this method makes the image to be output of mtcnn of exact size of 160 by 160
    '''

    img = face_detection(img)

    while img.shape[0] != destination_image_dim or img.shape[1] != destination_image_dim:
        img = img_resize_pil(img, destination_image_dim)
        img = np.asarray(img, 'uint8')
        img = face_detection(img)

    return img
