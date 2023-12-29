from fused_models.FusedModelBaseClass import FusedModel

import matplotlib.pyplot as plt

import numpy as np


class FusedModelAutoEncoder(FusedModel):
    def __init__(self, model_name, generator, face_descriptor, loss_function, X_train, img_size,
                 img_unnormalizer_function, epochs, batch_size, save_interval):

        # the average of all embeddings into (1,512)
        self.avg_embedding = np.mean(X_train, axis=0)

        # useless
        latent_dim = face_descriptor.output_shape[1]

        super().__init__(model_name, generator, face_descriptor, loss_function, latent_dim, X_train, img_size,
                         img_unnormalizer_function, epochs, batch_size, save_interval)

    def train(self):
        for epoch in range(1, self.epochs + 1):

            batch = np.repeat(self.avg_embedding, self.batch_size, axis=0)

            loss = self.combined.train_on_batch(batch, batch)

            print("%d [ loss: %f]" % (epoch, loss))
            self.write_to_csv(epoch, loss)

            if epoch % self.save_interval == 0 or epoch == 1:
                self.save_imgs(epoch)

        self.generator.save(self.dir_weights)
        self.make_gif()

    def save_imgs(self, epoch):
        gen_imgs = self.generator.predict(self.avg_embedding)

        gen_imgs = self.img_unnormalizer_function(gen_imgs)

        plt.imshow(gen_imgs[0].reshape(self.img_size, self.img_size, 3).astype(np.uint8))
        plt.savefig(self.dir_images + '/iter_%06d.png' % epoch)
        plt.close()
