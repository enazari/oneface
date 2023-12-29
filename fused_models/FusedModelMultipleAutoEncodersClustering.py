from fused_models.FusedModelBaseClass import FusedModel
import os

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import csv
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

def l2norm(vec):
    return vec / np.sqrt(np.sum(vec ** 2, -1, keepdims=True))


from PIL import Image
def img_resize_pil(img, output_dim):#, resample=Image.LANCZOS):
    '''
    The resize method used a default value for resampling which is now set to LANCZOS.
    Not passing the resample parameter will activate its default value.
    This change is made to make iterative use of MTCNN possible. '''
    image = Image.fromarray(img)
    image = image.resize(size=(output_dim, output_dim), resample=Image.BOX) #if set to LANCZOS instead of BOX, the coverage will drop (first observed defense)
    image = np.asarray(image, 'float64')
    return image

# import cv2
# def img_resize_cv2(img, output_dim):
#     img = cv2.resize(img, (output_dim, output_dim))
#     img = img.astype('float64')
#     return img






class FusedModelMultipleAutoEncodersClustering(FusedModel):
    '''
    The models which are input to this class need to have their final L2 normalization layer dropped(if they have one).
    L2 normalization is being controlled via a paramter named apply_l2
    '''
    def __init__(self, model_name, generator, face_descriptor, loss_function, img_size,
                 img_unnormalizer_function, img_normalizer_function, epochs, batch_size, save_interval,
                 number_of_clusters, apply_l2, where_to_obtain_centroids,
                 distance_func,
                 number_of_data_for_dev_purpose_only=None,
                 target_cluster=-1 ):

        self.threshold_for_best_accuracy = None
        assert where_to_obtain_centroids in ['all', 'identities', 'remaining'], 'where_to_obtain_centroids needs to be one of the following: all, identity, or remaining'


        self.where_to_obtain_centroids = where_to_obtain_centroids
        self.number_of_data_for_dev_purpose_only = number_of_data_for_dev_purpose_only #if equal to none, the algorithm is run for all data, if an integer, the algo goes through the first int data.
        self.the_rest_indices = None
        self.one_img_per_identity_indices = None
        self.closet_threshold_to_FAR_001 = None
        self.closet_threshold_to_FPR_001 = None
        self.best_accuracy_index_among_closet_FAR_to_001 = None
        self.thresh_stack = None
        self.frr_stack = None
        self.far_stack = None
        self.acc_stack = None
        self.apply_l2 = apply_l2
        latent_dim = face_descriptor.output_shape[1]

        if self.apply_l2 == True:
            output = Lambda(lambda x: K.l2_normalize(x, axis=1), name='added_lambda')(face_descriptor.output)
            face_descriptor = Model(inputs=face_descriptor.inputs, outputs=output)

        X_train=None # A value will be assigned to it in the following lines
        super().__init__(model_name, generator, face_descriptor, loss_function, latent_dim, X_train, img_size,
                         img_unnormalizer_function, epochs, batch_size, save_interval)


        #2200 pairs:
        self.train_faces = np.load('datasets/lfw_train_125_94_funneled_pairs.npz')['data']
        #1000 pairs:
        self.test_faces = np.load('datasets/lfw_test_125_94_funneled_pairs.npz')['data']

        #13233 faces:
        self.all_faces = np.load('datasets/people_lfw_125_by_94.npz')['data']
        self.all_faces_labels = np.load('datasets/people_lfw_125_by_94.npz')['label']
        self.divide_into_one_img_per_identity_and_the_rest() #should be called after self.all_face_labels are initialized




        self.X_train = self.return_embeddings(data=self.train_faces,
                                              face_descriptor=face_descriptor,
                                              preprocessor=img_normalizer_function,
                                              network_input_img_dim=self.img_size)

        self.X_test = self.return_embeddings(data=self.test_faces,
                                              face_descriptor=face_descriptor,
                                              preprocessor=img_normalizer_function,
                                              network_input_img_dim=self.img_size)

        self.X_all_faces = self.return_embeddings(data=self.all_faces,
                                              face_descriptor=face_descriptor,
                                              preprocessor=img_normalizer_function,
                                              network_input_img_dim=self.img_size)
        ''''''


        #used for masterface testing:
        if self.number_of_data_for_dev_purpose_only == None:
            self.X_one_face_per_identity = self.X_all_faces[self.one_img_per_identity_indices]
            self.X_the_rest_faces = self.X_all_faces[self.the_rest_indices]
        else:
            self.X_one_face_per_identity = self.X_all_faces
            self.X_the_rest_faces = self.X_all_faces

        # temporary:

        # folds  = self.return_embeddings(data=np.load('datasets/lfw_10folds_125_94_funneled_pairs.npz')['data'],
        #                                       face_descriptor=face_descriptor,
        #                                       preprocessor=img_normalizer_function,
        #                                       network_input_img_dim=self.img_size)
        # np.savez_compressed('datasets/lfw_train_facenet_casia_embeddings', data=self.X_train)
        # np.savez_compressed('datasets/lfw_test_facenet_casia_embeddings', data=self.X_test)
        # np.savez_compressed('datasets/lfw_all_faces_facenet_casia_embeddings', data=self.X_all_faces)
        # np.savez_compressed('datasets/lfw_one_face_per_identity_facenet_casia_embeddings', data=self.X_one_face_per_identity)
        # np.savez_compressed('datasets/lfw_the_rest_faces_facenet_casia_embeddings', data=self.X_the_rest_faces)
        #
        # np.savez_compressed('datasets/xcxcxc', the_rest_faces=self.X_the_rest_faces,
        #                     one_face_per_identity=self.X_one_face_per_identity,
        #                     all_faces=self.X_all_faces,
        #                     test=self.X_test,
        #                     train=self.X_train)#,folds = folds.reshape(6000,2,1,512))
        #:temporary




        self.results_csv_file_name = '_coverages.csv'
        self.threshold_results_csv_file_name = '_thresholds.csv'

        self.img_normalizer_function = img_normalizer_function
        self.middle_threshold = None
        self.distance_func = distance_func
        self.obtain_threshold(data=self.X_train)

        #the following lines must be called after calling self.obtain_threshold() and in order:
        self.calc_metrics_in_a_stack()
        self.obtain_best_001_far()
        self.obtain_best_001_fpr()
        self.save_thresholds()
        #:the following lines must be called after calling self.obtain_threshold() and in order


        self.target_cluster = target_cluster


        '''
        13233      (all lfw people faces)                           --> name: all
        5749       (identities - used for master face coverage)     --> name: identities
        7484       ( the remaining images used to obtain centroids) --> name: remaining'''
        self.clusters_centers = None
        self.number_of_clusters = number_of_clusters
        if self.where_to_obtain_centroids == 'all':
            self.return_centers_of_clusters(data=self.X_all_faces)
        elif self.where_to_obtain_centroids == 'identities':
            self.return_centers_of_clusters(data=self.X_one_face_per_identity)
        elif self.where_to_obtain_centroids == 'remaining':
            self.return_centers_of_clusters(data=self.X_the_rest_faces)

    def divide_into_one_img_per_identity_and_the_rest(self):
            one_img_per_identity_set = set()
            one_img_per_identity_indices = list()
            the_rest_indices = list()

            for index, person in enumerate(self.all_faces_labels):
                if person in one_img_per_identity_set:
                    the_rest_indices.append(index)
                else:
                    one_img_per_identity_set.add(person)
                    one_img_per_identity_indices.append(index)
            self.one_img_per_identity_indices = one_img_per_identity_indices
            self.the_rest_indices = the_rest_indices
            return None

    def return_embeddings(self, data, face_descriptor, preprocessor, network_input_img_dim):
        data = data.reshape(-1, data.shape[-3], data.shape[-2],
                            data.shape[-1])  # print(data.shape) returns (-1, 125,94,3)
        embeddings = []
        for index, img in enumerate(data[:self.number_of_data_for_dev_purpose_only]):#data[:4]:

            img = img_resize_pil(img, network_input_img_dim)
            # img = img_resize_cv2(img, network_input_img_dim)

            img = img.reshape((1,) + img.shape)
            preprocessed = preprocessor(img)
            embeddings.append(face_descriptor(preprocessed))

        embeddings_np = np.array(embeddings)
        embeddings_np = embeddings_np.reshape(-1,1, embeddings_np.shape[-1])
        return embeddings_np

    def return_centers_of_clusters(self, data):

        kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=0).fit(data.reshape(-1, data.shape[-1]))
        cluster_labels = kmeans.labels_

        cluster_centers = []
        set_of_labels = np.unique(cluster_labels)
        for label in set_of_labels:
            cluster = data[label == cluster_labels]
            m = np.mean(cluster, axis=0)
            if self.apply_l2 == True:
                m = l2norm(m)
            cluster_centers.append(m)
        self.cluster_centers = cluster_centers

    def obtain_threshold(self, data):
        '''
        the assumption for given data is that it contains pairs, FIRST HALF of which are of matching,
        and SECOND HALF of which are mismatching.
        '''
        data = data.reshape(-1,2, data.shape[-1])
        sames = []
        diffs = []
        same_diff_point = int(len(data)/2)
        for index, d in enumerate(data):

            #if the given datum is of class same
            if index < same_diff_point:
                sames.append( self.distance_func(d[0], d[1]) )
            else:
                diffs.append( self.distance_func(d[0], d[1]) )
        same_average = sum(sames) / len(sames)
        diff_average = sum(diffs) / len(diffs)
        threshold = (same_average + diff_average)/2

        self.middle_threshold = threshold
        return None

    def master_tester(self, potential_master_embedding, data):

        data = data.reshape(-1, 1, data.shape[-1])

        opened = []
        for d in data:
            if self.distance_func(d, potential_master_embedding) < self.middle_threshold:
                opened.append(1)
            else:
                opened.append(0)

        return opened

    def accumulative_coverage_calc_helper(self, unlockeds): #TODO: if unlockeds is empty, should we return 0?
        unlockeds = np.array(unlockeds)
        if unlockeds.shape[-1] == 0:
            return 0

        accumulation = np.any(unlockeds, axis=0)
        global_coverage = round(np.count_nonzero(accumulation)*100/accumulation.shape[0], 3)
        return global_coverage

    def local_coverage_calc_helper(self, unlockeds):
        count = len(unlockeds)
        if count == 0:
            return 0
        local_coverage = sum(unlockeds) / len(unlockeds)
        local_coverage = round(100 * local_coverage, 3)
        return local_coverage


    def save_to_csv_local(self, unlocked_direct, unlocked_indirect, clstr_num):

        direct_local = self.local_coverage_calc_helper(unlocked_direct)
        indirect_local = self.local_coverage_calc_helper(unlocked_indirect)


        cols = ['cluster',
                'direct local coverage percent',
                'indirect local coverage percent',
                'direct global coverage percent',
                'indirect global coverage percent',
                'member count',
                'L2 normalized',
                'dataset']
        row=[str(clstr_num+1)+' of '+str(len(self.cluster_centers)),
                          direct_local,
                          indirect_local,
                          '-',
                          '-',
                          str(len(unlocked_direct)),
                          self.apply_l2,
                          self.model_name ]

        file_path = self.dir_config+'/'+self.results_csv_file_name
        file_exists = os.path.exists(file_path)
        with open(file_path, 'a') as csv_file:
            writer = csv.writer(csv_file)
            #if a new csv file is being created:
            if file_exists == False:
                writer.writerow(cols)

            writer.writerow(row)


    def save_to_csv_global(self, unlockeds_direct, unlockeds_indirect):

        direct_global_coverage = self.accumulative_coverage_calc_helper(unlockeds_direct)
        indirect_global_coverage = self.accumulative_coverage_calc_helper(unlockeds_indirect)

        row = ['accumulation of '+str(len(self.cluster_centers)),
                          '-',
                          '-',
                          direct_global_coverage,
                          indirect_global_coverage,
                          str(np.any(np.array(unlockeds_direct), axis=0).shape[0]),
                          self.apply_l2,
                          self.model_name ]

        file_path = self.dir_config+'/'+self.results_csv_file_name
        with open(file_path, 'a') as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(row)


    def train(self):
        unlucked_test_cases_for_cluster_centers=[]
        unlucked_test_cases_for_reconstructed_cluster_centers=[]

        for index, cluster_center in enumerate(self.cluster_centers):
            #training on the given cluster only: (activated when target_cluster is not -1)
            if self.target_cluster != -1:
                if self.target_cluster != index+1:
                    continue

            #BAD Practice- creating new directories:
            cluster_name = '_cluster'+str(index+1)
            dir_imgs = self.dir_images + cluster_name
            dir_wghts = self.dir_weights + cluster_name
            for dir in [dir_imgs, dir_wghts]:
                if not os.path.exists(dir):
                    os.makedirs(dir)

            #Calculating the coverage of the centroids================================:
            unlucked_test_cases_for_this_cluster_direct = self.master_tester(cluster_center, data=self.X_one_face_per_identity)
            unlucked_test_cases_for_cluster_centers.append(unlucked_test_cases_for_this_cluster_direct)
            #:Calculating the coverage of the centroids================================


            for epoch in range(1, self.epochs + 1):

                batch = np.repeat(cluster_center, self.batch_size, axis=0)

                loss = self.combined.train_on_batch(batch, batch)

                print("%d [ loss: %f]" % (epoch, loss))
                self.write_to_csv(epoch, loss)

                if epoch % self.save_interval == 0 or epoch == 1:
                    self.save_imgs(epoch, cluster_center, dir_imgs)

            self.generator.save(dir_wghts)
            # self.make_gif()

            #Calculating the coverage of the recunstructed centroids==================:
            gen_img = self.generator.predict(cluster_center)
            gen_img_unnormalized=  self.img_unnormalizer_function(gen_img)
            gen_img_unnormalized = gen_img_unnormalized[0].reshape(1, self.img_size, self.img_size, 3).astype(np.uint8)
            gen_img_unnormalized = gen_img_unnormalized.astype('float64')
            gen_img_normalized = self.img_normalizer_function(gen_img_unnormalized)
            reconstructed_cluster_center = self.face_descriptor(gen_img_normalized)
            reconstructed_cluster_center = reconstructed_cluster_center.numpy().reshape(1, -1)

            unlucked_test_cases_for_this_cluster_indirect = self.master_tester(reconstructed_cluster_center, data=self.X_one_face_per_identity)
            unlucked_test_cases_for_reconstructed_cluster_centers.append(unlucked_test_cases_for_this_cluster_indirect)
            #:Calculating the coverage of the recunstructed centroids==================

            self.save_to_csv_local(unlocked_direct=unlucked_test_cases_for_this_cluster_direct,
                                   unlocked_indirect=unlucked_test_cases_for_this_cluster_indirect,
                                   clstr_num=index)

        #saving accumulative results:
        self.save_to_csv_global(unlucked_test_cases_for_cluster_centers, unlucked_test_cases_for_reconstructed_cluster_centers)

    def save_imgs(self, epoch, cluster_center, dir_imgs):
        gen_imgs = self.generator.predict(cluster_center)

        gen_imgs = self.img_unnormalizer_function(gen_imgs)

        plt.imshow(gen_imgs[0].reshape(self.img_size, self.img_size, 3).astype(np.uint8))
        plt.savefig(dir_imgs + '/iter_%06d.png' % epoch)
        plt.close()

    # helper functions to obtain the closest threshold to false acceptance rate of 0.001:

    @staticmethod
    def accuracy(tp, fp, tn, fn):
        return (tp + tn) / (tp + fp + tn + fn)
    @staticmethod
    def false_acceptance_rate(tp, fp, tn, fn):
        return fp / (tp + fp + tn + fn)

    @staticmethod
    def false_positive_rate(tp, fp, tn, fn):
        return fp / (fp + tn)
    @staticmethod
    def false_rejection_rate(tp, fp, tn, fn):
        return fn / (tp + fp + tn + fn)
    def metrics(self, pairs, thresh):
        '''
        :param pairs: the first half of the data needs to be mactching pairs.
        the second half of data needs to be mismatching pairs
        '''
        tp, fp, tn, fn = 0, 0, 0, 0
        same_diff_point = int(len(pairs) / 2)
        for index, d in enumerate(pairs):
            # if the given datum is of class same
            if index < same_diff_point:
                if self.distance_func(d[0], d[1]) < thresh:
                    tp += 1
                else:
                    fn += 1
            else:
                if self.distance_func(d[0], d[1]) > thresh:
                    tn += 1
                else:
                    fp += 1
        return tp, fp, tn, fn

    def calc_metrics(self, pairs, thresh):
        tp, fp, tn, fn = self.metrics(pairs, thresh)
        acc = self.accuracy(tp, fp, tn, fn)
        far = self.false_acceptance_rate(tp, fp, tn, fn)
        frr = self.false_rejection_rate(tp, fp, tn, fn)

        fpr = self.false_positive_rate(tp, fp, tn, fn)

        return acc, far, frr, fpr
    # :helper functions to obtain the closest threshold to false acceptance rate of 0.001

    def calc_metrics_in_a_stack(self):
        pairs = self.X_train.reshape(-1,2, self.X_train.shape[-1])

        acc_stack, far_stack, frr_stack, thresh_stack, fpr_stack = list(), list(), list(), list(), list()

        threshes = np.linspace(start=self.middle_threshold * 0.8, stop=self.middle_threshold * 1.2, num=100)
        threshes = np.append(threshes, self.middle_threshold)
        for i in threshes:
            acc, far, frr, fpr = self.calc_metrics(pairs, i)
            acc_stack.append(acc)
            far_stack.append(far)
            fpr_stack.append(fpr)
            frr_stack.append(frr)
            thresh_stack.append(i)

        self.acc_stack = acc_stack
        self.far_stack = far_stack
        self.fpr_stack = fpr_stack
        self.frr_stack = frr_stack
        self.thresh_stack = thresh_stack
        return None#acc_stack, far_stack, frr_stack, thresh_stack

    def obtain_best_001_far(self):

        # finding the best accuracy among FARs that are the closest to 0.001:
        far_stack_np = np.array(self.far_stack)
        far_stack_np = far_stack_np - 0.001
        far_dist_to_001 = abs(far_stack_np)

        min_indices = np.where(far_dist_to_001 == far_dist_to_001.min())

        temp_acc_stack = np.array([[value, index] for index, value in enumerate(self.acc_stack)])

        local_index = np.argmax(temp_acc_stack[min_indices][:, 0], axis=0)

        best_accuracy_index_among_closet_FAR_to_001 = int(temp_acc_stack[min_indices][local_index][1])
        self.best_accuracy_index_among_closet_FARs_to_001 = best_accuracy_index_among_closet_FAR_to_001
        self.closet_threshold_to_FAR_001 = self.thresh_stack[best_accuracy_index_among_closet_FAR_to_001]

        return None #best_accuracy_index_among_closet_FARs_to_001, self.thresh_stack[best_accuracy_index_among_closet_FARs_to_001]

    def obtain_best_001_fpr(self):

        # finding the best accuracy among FARs that are the closest to 0.001:
        fpr_stack_np = np.array(self.fpr_stack)
        fpr_stack_np = fpr_stack_np - 0.001
        fpr_dist_to_001 = abs(fpr_stack_np)

        min_indices = np.where(fpr_dist_to_001 == fpr_dist_to_001.min())

        temp_acc_stack = np.array([[value, index] for index, value in enumerate(self.acc_stack)])

        local_index = np.argmax(temp_acc_stack[min_indices][:, 0], axis=0)

        best_accuracy_index_among_closet_FPR_to_001 = int(temp_acc_stack[min_indices][local_index][1])
        self.best_accuracy_index_among_closet_FPRs_to_001 = best_accuracy_index_among_closet_FPR_to_001
        self.closet_threshold_to_FPR_001 = self.thresh_stack[best_accuracy_index_among_closet_FPR_to_001]
    def save_thresholds(self):

        cols = ['name',
                'threhold',
                'false acceptance rate',
                'false rejection rate',
                'accuracy',
                'false positive rate',
                ]
        rows=[]

        golden_index = self.best_accuracy_index_among_closet_FARs_to_001
        rows.append([
                          'Nearest FAR to 0.001',
                          self.thresh_stack[golden_index],
                          self.far_stack[golden_index],
                          self.frr_stack[golden_index],
                          self.acc_stack[golden_index],
                          self.fpr_stack[golden_index]
        ])


        golden_index = self.best_accuracy_index_among_closet_FPRs_to_001
        rows.append([
                          'Nearest FPR to 0.001',
                          self.thresh_stack[golden_index],
                          self.far_stack[golden_index],
                          self.frr_stack[golden_index],
                          self.acc_stack[golden_index],
                          self.fpr_stack[golden_index]
        ])


        best_accuracy_index = np.argmax(np.array(self.acc_stack))
        rows.append([
                          'Best training-set accuracy',
                          self.thresh_stack[best_accuracy_index],
                          self.far_stack[best_accuracy_index],
                          self.frr_stack[best_accuracy_index],
                          self.acc_stack[best_accuracy_index],
                          self.fpr_stack[best_accuracy_index]
                     ])
        self.threshold_for_best_accuracy = self.thresh_stack[best_accuracy_index]

        middle_thresh_index = np.where(np.array(self.thresh_stack) == self.middle_threshold)
        rows.append([
                          'middle threshold',
                          self.thresh_stack[middle_thresh_index[0][0]],
                          self.far_stack[middle_thresh_index[0][0]],
                          self.frr_stack[middle_thresh_index[0][0]],
                          self.acc_stack[middle_thresh_index[0][0]],
                          self.fpr_stack[middle_thresh_index[0][0]]
        ])


        file_path = self.dir_config+'/'+self.threshold_results_csv_file_name
        file_exists = os.path.exists(file_path)
        with open(file_path, 'a') as csv_file:
            writer = csv.writer(csv_file)
            #if a new csv file is being created:
            if file_exists == False:
                writer.writerow(cols)

            for row in rows:
                writer.writerow(row)

