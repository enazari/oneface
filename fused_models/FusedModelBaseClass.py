from fused_models.BaseClass import BaseClass

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

import matplotlib.pyplot as plt

from csv import writer

import numpy as np

import glob
import imageio


class FusedModel(BaseClass):
    def __init__(self,
                 model_name,
                 generator,
                 face_descriptor,
                 loss_function,
                 latent_dim,
                 X_train,
                 img_size,
                 img_unnormalizer_function,
                 epochs,
                 batch_size,
                 save_interval):

        self.generator = generator
        self.latent_dim = latent_dim
        self.save_interval = save_interval

        # noise for generated images:
        self.r, self.c = 5, 5  # number of rows and columns of generated images
        self.noise_to_generate_imgs = np.random.normal(0, 1, (self.r * self.c, self.latent_dim))

        super().__init__(model_name,
                         face_descriptor,
                         loss_function,
                         X_train,
                         img_size,
                         img_unnormalizer_function,
                         epochs,
                         batch_size,
                         save_interval)

        self.combined = self.return_combined_model()


    def return_combined_model(self):
        optimizer = Adam(0.0002, 0.5)
        z = Input(shape=(self.latent_dim,))
        self.generator._name= 'generator_model' #added to prevent an error caused when face_descriptor is facenet and l2 is False
        img = self.generator(z)

        self.face_descriptor.trainable = False

        embedding = self.face_descriptor(img)

        combined = Model(z, embedding, name='combined_model') #name added to prevent an error caused when face_descriptor is facenet and l2 is False
        combined.compile(loss=self.loss_function, optimizer=optimizer)

        return combined

    def train(self):
        for epoch in range(1, self.epochs + 1):

            idx = np.random.randint(0, self.X_train.shape[0], self.batch_size)
            imgs = self.X_train[idx]

            noise = np.random.normal(0, 1, (1, self.latent_dim))

            noise = np.repeat(noise, self.batch_size, axis=0)

            g_loss = self.combined.train_on_batch(noise, imgs)

            print("%d  [G loss: %f]" % (epoch, g_loss))
            self.write_to_csv(epoch, g_loss)

            if epoch % self.save_interval == 0 or epoch == 1:
                self.save_imgs(epoch)

        self.generator.save(self.dir_weights)
        self.make_gif()

    def save_imgs(self, epoch):
        r, c = self.r, self.c
        gen_imgs = self.generator.predict(self.noise_to_generate_imgs)

        gen_imgs = self.img_unnormalizer_function(gen_imgs)
        # gen_imgs[..., 0] += 91.4953
        # gen_imgs[..., 1] += 103.8827
        # gen_imgs[..., 2] += 131.0912
        # gen_imgs = gen_imgs[..., ::-1]

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt].reshape(self.img_size, self.img_size, 3).astype(np.uint8))
                axs[i, j].axis('off')
                cnt += 1

        fig.set_figheight(15)
        fig.set_figwidth(15)
        fig.savefig(self.dir_images + '/iter_%06d.png' % epoch)
        plt.close()

    def make_gif(self):
        anim_file = self.dir_images + '/_the_gif.gif'
        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(self.dir_images + '/iter*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.v2.imread(filename)
                writer.append_data(image)
            image = imageio.v2.imread(filename)
            writer.append_data(image)

    def write_to_csv(self, epoch, loss):

        with open(self.dir_loss_history + '/loss_history.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([epoch, loss])
            f_object.close()

# if __name__ == '__main__':
#     from keras_vggface.vggface import VGGFace
#     from loss_functions.euclidean import euclidean_distance_loss
#     from generator_models.regular256 import return_generator_model
#     img_size = 256
#     VGG = VGGFace(model='vgg16', include_top=False, input_shape=(img_size, img_size, 3), pooling='avg')
#
#     X_train = np.load('../datasets/X_embedding_VGG16_imgsize256.npz')['X_embedding']
#     Masterkey1 = FusedModel(model_name = 'regular256',
#                            generator = return_generator_model(),
#                            face_descriptor = VGG,
#                            loss_function = euclidean_distance_loss,
#                            latent_dim = 1024,
#                            X_train = X_train,
#                            img_size= img_size)
#     Masterkey1.train()
