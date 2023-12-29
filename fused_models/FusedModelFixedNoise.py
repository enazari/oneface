from fused_models.FusedModelBaseClass import FusedModel as VanillaFusedModel

import numpy as np
class FusedModel(VanillaFusedModel):
    def __init__(self, model_name, generator, face_descriptor, loss_function, latent_dim, X_train, img_size,
                 img_unnormalizer_function, epochs, batch_size, save_interval):
        super().__init__(model_name, generator, face_descriptor, loss_function, latent_dim, X_train, img_size,
                         img_unnormalizer_function, epochs, batch_size, save_interval)


    def train(self):
        #If a huge batch size is set, then it will become of oneThid the training size:
        if self.batch_size > self.X_train.shape[0]:
            self.batch_size = int( self.X_train.shape[0]/4)

        noise = np.random.normal(0, 1, (1, self.latent_dim))

        noise = np.repeat(noise, self.batch_size, axis=0)

        for epoch in range(1, self.epochs + 1):

            idx = np.random.randint(0, self.X_train.shape[0], self.batch_size)
            imgs = self.X_train[idx]

            g_loss = self.combined.train_on_batch(noise, imgs)

            print("%d  [G loss: %f]" % (epoch, g_loss))
            self.write_to_csv(epoch, g_loss)

            if epoch % self.save_interval == 0 or epoch == 1:
                self.save_imgs(epoch)

        self.generator.save(self.dir_weights)
        self.make_gif()