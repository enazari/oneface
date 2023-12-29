

import os

from datetime import datetime

from csv import writer


import glob
import imageio


class BaseClass:
    def __init__(self,
                 model_name,
                 face_descriptor,
                 loss_function,
                 X_train,
                 img_size,
                 img_unnormalizer_function,
                 epochs,
                 batch_size,
                 save_interval):

        self.model_name = model_name
        self.face_descriptor = face_descriptor
        self.loss_function = loss_function
        self.X_train = X_train
        self.img_size = img_size
        self.img_unnormalizer_function = img_unnormalizer_function
        self.save_interval = save_interval
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_shape = (self.img_size, self.img_size, 3)

        now = datetime.now()

        model_folder_name = self.model_name + '_' + now.strftime("%d_%m_%Y_%H_%M_%S.%f")
        self.dir_weights = 'results/' + model_folder_name + '/weights'
        self.dir_images = 'results/' + model_folder_name + '/images'
        self.dir_config = 'results/' + model_folder_name
        self.dir_loss_history = self.dir_config
        for dir in [self.dir_weights,
                    self.dir_images,
                    self.dir_config,
                    self.dir_loss_history]:
            if not os.path.exists(dir):
                os.makedirs(dir)



    def train(self):
        pass

    def save_imgs(self, epoch):
        pass

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
