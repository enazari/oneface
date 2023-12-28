#trained on CASIA-WebFace Dataset 20180408-102900 :  https://github.com/davidsandberg/facenet
# embedder = FaceNet(key='20180408-102900')
from keras_facenet import FaceNet
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np

def input_generated_imgs_output_unnormalized_imgs(gen_imgs):
    return (np.float32(gen_imgs) * 127.5) + 127.5

def unnormalizer(imgs):
    return input_generated_imgs_output_unnormalized_imgs(imgs)

#last layer (which is a l2-normalization layer) is omited:
def return_model():

    #method1:
    # embedder = FaceNet()
    # truncated_facenet = Model(inputs=embedder.model.inputs, outputs=embedder.model.layers[-2].output)

    #method2:
    import os
    print( os.path.abspath(os.getcwd()) )

    truncated_facenet = load_model('face_descriptors/facenet_l2_dropped_trained_on_casia_saved_model')

    return truncated_facenet

def preprocess_input(imgs):
    return (np.float32(imgs) - 127.5) / 127.5

def return_normalizer():
    return preprocess_input


def return_model_and_unnormalizer():
    model = return_model()
    return model, input_generated_imgs_output_unnormalized_imgs


if __name__ == '__main__':
    facenet = return_model()
    facenet.summary()
