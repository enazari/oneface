from keras_vggface.vggface import VGGFace

from keras_vggface.utils import preprocess_input

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model


def input_generated_imgs_output_unnormalized_imgs(gen_imgs): #THESE NUMBERS ARE SLIGHTLY WRONG!
    gen_imgs[..., 0] += 91.4953
    gen_imgs[..., 1] += 103.8827
    gen_imgs[..., 2] += 131.0912
    gen_imgs = gen_imgs[..., ::-1]
    return gen_imgs

def input_generated_imgs_output_unnormalized_imgs2(gen_imgs):
    gen_imgs[..., 0] += 93.5940
    gen_imgs[..., 1] += 104.7624
    gen_imgs[..., 2] += 129.1863
    gen_imgs = gen_imgs[..., ::-1]
    return gen_imgs

def unnormalizer(imgs):
    return input_generated_imgs_output_unnormalized_imgs(imgs)

def return_model(img_size, pooling='avg'):
    vgg = VGGFace(model='vgg16', include_top=True, input_shape=(img_size, img_size, 3), pooling=pooling)

    embedding_model = Model(inputs=vgg.inputs, outputs=vgg.layers[-3].output)

    return embedding_model


def return_normalizer():
    return preprocess_input


def return_model_and_unnormalizer(img_size, pooling='avg'):
    vgg16 = return_model(img_size, pooling)
    return vgg16, input_generated_imgs_output_unnormalized_imgs


if __name__ == '__main__':
    vgg16 = return_model(224)
    vgg16.summary()