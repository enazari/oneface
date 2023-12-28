
from keras_vggface.utils import preprocess_input
from tensorflow.keras.models import load_model, Model
import  tensorflow as tf
def input_generated_imgs_output_unnormalized_imgs(gen_imgs):
    gen_imgs[..., 0] += 93.5940
    gen_imgs[..., 1] += 104.7624
    gen_imgs[..., 2] += 129.1863
    gen_imgs = gen_imgs[..., ::-1]
    return gen_imgs

def unnormalizer(imgs):
    return input_generated_imgs_output_unnormalized_imgs(imgs)

def return_model(img_size):
    assert img_size == 224
    embedding_model = load_model('other/triplet_loss_training_of_vgg16/results/img_dim_224_embed_type_vgg_data_train_batch_128_epochs_10_test_fold_-1_alpha_1_R876_embeddingModel_fc8/model')

    for layer in embedding_model.layers:
        layer.trainable = False


    return embedding_model


def return_normalizer():
    return preprocess_input


def return_model_and_unnormalizer(img_size):
    vgg16 = return_model(img_size)
    return vgg16, input_generated_imgs_output_unnormalized_imgs


if __name__ == '__main__':
    vgg16 = return_model(224)
    vgg16.summary()