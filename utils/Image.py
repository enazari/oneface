from PIL import Image
import numpy as np


def img_resize_pil(img, output_dim, resample=Image.BOX):
    '''
    The resize method used a default value for resampling which is now set to BOX.
    Not passing the resample parameter will activate its default value.
    This change is made to make iterative use of MTCNN possible. '''

    image = Image.fromarray(img)
    image = image.resize(size=(output_dim, output_dim),
                         resample=resample)  # if set to LANCZOS instead of BOX, the coverage will drop (first observed defense)
    image = np.asarray(image, 'float64')
    return image


def clip_to_0_255_and_change_dtype_to_uint8(img):
    return np.clip(img, a_min=0, a_max=255).astype(np.uint8)


def read_image(path):
    img = np.array(Image.open(path))
    return img

def enfore_uint8_on_image(img):
    img = img[:,:,:3]
    if np.max(img) < 2 : # if [0,1], convert to [0,255]
        img *= 255
        img = clip_to_0_255_and_change_dtype_to_uint8(img)
    return img

def get_image(path):
    img = read_image(path)
    return  enfore_uint8_on_image(img)

def facenet_preprocessing( image):
    return (np.float32(image) - 127.5) / 127.5
