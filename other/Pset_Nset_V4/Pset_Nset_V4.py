from fused_models.FusedModelMultipleAutoEncodersClustering import *
import tensorflow as tf
from skimage.metrics import structural_similarity as compare_ssim

import time


from GA.opti.lmmaes import Lmmaes
from GA.prob.problems import Weighted_PSet_NSet_LockAndDistance

from copy import deepcopy
import random
import numpy as np
def euclidean_np(a, b):
    return np.linalg.norm(a-b)
def cosine_np(a,b):
    return 1 - np.sum(a*b)/(np.linalg.norm(a) * np.linalg.norm(b))

# the following is obtain from https://github.com/davidsandberg/facenet/blob/096ed770f163957c1e56efa7feeb194773920f6e/src/facenet.py#L101
# it is based on cosine similarity:
import math
def cosine_variant(a, b):
    dot = np.sum(np.multiply(a, b))
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    similarity = dot / norm
    return np.arccos(similarity) / math.pi


from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

from mtcnn.mtcnn import MTCNN
_mtcnn = MTCNN()




class Pset_Nset_V4(FusedModelMultipleAutoEncodersClustering):
    '''
    Here we have 4 list inputs:
    seed_list
    p_set_number_of_identities_list
    p_set_number_of_clusters_list
    n_set_number_of_identities_list
    '''
    def __init__(self, model_name,
                 face_descriptor,
                 img_size,
                 img_unnormalizer_function,
                 img_normalizer_function,
                 apply_l2,
                 number_of_data_for_dev_purpose_only,
                 distance_func, # of type string
                 use_threshold_for_best_accuracy_or_for_closest_to_FAR001,
                 p_set_threshold_coefficient,
                 n_set_threshold_coefficient,

                 seed_list,

                 #Genetic Algo params:
                 GA_number_of_generations,
                 GA_sigma,
                 GA_population_size,

                 p_set_number_of_identities_list,
                 p_set_instances_per_identity, # the number of instances per identity for p_set is equal to pstar_set instances plus pure p_set instances
                 pstar_set_instances_per_identity, #pset and pstar_set are disjoint set of the same identites where pstar_set is not seen during the training
                 p_set_number_of_clusters_list,
                 p_data_lock_weight,

                 n_set_number_of_identities_list,
                 n_set_instances_per_identity,
                 nstar_set_instances_per_identity, #nstar_set is a subset of n_set, where is not seen during the training
                 n_data_lock_weight,

                 p_n_balance_weight,



                # derive image params:
                deriveImg_a_face,
                use_the_only_nset_image_as_deriveImg,
                deriveImg_epsilon,
                deriveImg_loss_function,
                deriveImg_iterations,
                clip_to_min,
                clip_to_max,
                deriveImg_perturbation_limit
                 ):
        self.start_time = time.time()

        #These is omited because it has default values : target_cluster = -1
        self.use_the_only_nset_image_as_deriveImg = use_the_only_nset_image_as_deriveImg
        p, pstar = p_set_instances_per_identity, pstar_set_instances_per_identity
        n, nstar = n_set_instances_per_identity, nstar_set_instances_per_identity
        assert pstar < p or pstar==p==0, 'pstar instances  per identity should be strictly smaller than p instances per identity'
        assert nstar < n or nstar==n==0, 'nstar instances  per identity should be strictly smaller than n instances per identity'
        self.nstar_set_instances_per_identity = nstar_set_instances_per_identity
        self.pstar_set_instances_per_identity = pstar_set_instances_per_identity

        self.deriveImg_perturbation_limit = deriveImg_perturbation_limit
        self.p_set_threshold_coefficient = p_set_threshold_coefficient
        self.n_set_threshold_coefficient = n_set_threshold_coefficient
        self.p_n_balance_weight = p_n_balance_weight
        self.n_data_lock_weight = n_data_lock_weight
        self.n_set_instances_per_identity = n_set_instances_per_identity
        self.n_set_number_of_identities_list = n_set_number_of_identities_list
        self.p_data_lock_weight = p_data_lock_weight
        self.p_set_number_of_clusters_list = p_set_number_of_clusters_list
        self.p_set_instances_per_identity = p_set_instances_per_identity
        self.p_set_number_of_identities_list = p_set_number_of_identities_list
        self.seed_list = seed_list
        assert use_threshold_for_best_accuracy_or_for_closest_to_FAR001 in ['best', 'far001', 'fpr001'] , 'GA_algo can only be best or far001 or fpr001'
        self.GA_population_size = GA_population_size
        self.GA_sigma = GA_sigma
        self.GA_number_of_generations = GA_number_of_generations
        self.ssim_list = list()


        assert distance_func in ['euclidean', 'cosine', 'cosine_variant'] , 'the distance function should be either the string cosine or euclidean or cosine_variant'
        if distance_func == 'euclidean':
            distance_func = euclidean_np
        if distance_func == 'cosine':
            distance_func = cosine_np
        if distance_func == 'cosine_variant':
            distance_func = cosine_variant

        # REDUNDANT:
        # imported for arbitrary parameters(useless):
        from generator_models import mlp160_tanh
        from loss_functions import euclidean
        #These are reduntant parameters which are not needed in this class:
        #They are set to arbitrary values
        generator, imgsize = mlp160_tanh.return_generator_model_and_imgsize(face_descriptor.layers[-1].output_shape[-1])
        loss_function = euclidean.loss
        epochs=-1
        batch_size=-1,
        save_interval=-1
        # :REDUNDANT


        super().__init__(
            model_name = model_name,
            generator = generator,
            face_descriptor= face_descriptor,
            loss_function = loss_function,
            img_size = img_size,
            img_unnormalizer_function = img_unnormalizer_function,
            img_normalizer_function = img_normalizer_function,
            epochs = epochs,
            batch_size = batch_size,
            save_interval = save_interval,
            number_of_clusters=1,#useless
            apply_l2 = apply_l2,
            where_to_obtain_centroids = 'remaining', #Useless
            number_of_data_for_dev_purpose_only=number_of_data_for_dev_purpose_only,
            target_cluster=-1,
            distance_func=distance_func)


        if use_threshold_for_best_accuracy_or_for_closest_to_FAR001 == 'far001':
            self.thresh_in_use = self.closet_threshold_to_FAR_001
        if use_threshold_for_best_accuracy_or_for_closest_to_FAR001 == 'fpr001':
            self.thresh_in_use = self.closet_threshold_to_FPR_001
        if use_threshold_for_best_accuracy_or_for_closest_to_FAR001 == 'best':
            self.thresh_in_use = self.threshold_for_best_accuracy

        self.n_set_threshold = self.thresh_in_use * self.n_set_threshold_coefficient
        self.p_set_threshold = self.thresh_in_use * self.p_set_threshold_coefficient

        self.deriveImg_a_face = self.preprocess_autoencoder_face_mtcnn_fit(deriveImg_a_face)
        self.deriveImg_epsilon = deriveImg_epsilon
        self.deriveImg_loss_function = deriveImg_loss_function
        self.deriveImg_iterations = deriveImg_iterations
        self.clip_to_min = clip_to_min
        self.clip_to_max = clip_to_max



        self.deriveImg_img_dir = self.dir_config + '/imgs_deriveImg'




    def create_Pset_Nset(self,
                         p_set_number_of_clusters,
                         p_set_number_of_identities,
                         n_set_number_of_identities,
                         seed):

        random.seed(seed)

        p_set_identities = self.select_I_identities_each_of_which_has_S_instances(
            number_of_identities=p_set_number_of_identities,
            instances_per_identity=self.p_set_instances_per_identity,
            forbidden_identites_list=[])  # empty list

        p_set = []
        p_set_chosen_identities_indices = []

        pstar_set = []
        pstar_set_chosen_identities_indices = []
        for identity_id, available_instance_indices_of_this_identity in p_set_identities.items():
            #p_set:
            indices = deepcopy(available_instance_indices_of_this_identity)
            for instance in range(self.p_set_instances_per_identity - self.pstar_set_instances_per_identity):
                # second of the two places to insert randomness:
                chosen_index = random.choice( indices )
                p_set_chosen_identities_indices.append(chosen_index)
                p_set.append(self.X_all_faces[ chosen_index ])
                indices.remove(chosen_index)

            #pstar_set:
            for instance in range(self.pstar_set_instances_per_identity):
                # second of the two places to insert randomness:
                chosen_index = random.choice( indices )
                pstar_set_chosen_identities_indices.append(chosen_index)
                pstar_set.append(self.X_all_faces[ chosen_index ])
                indices.remove(chosen_index)

        p_set = np.array(p_set)
        pstar_set = np.array(pstar_set)


        n_set_identities = self.select_I_identities_each_of_which_has_S_instances(
            number_of_identities=n_set_number_of_identities,
            instances_per_identity=self.n_set_instances_per_identity,
            forbidden_identites_list=list(
                p_set_identities.keys()))  # whatever is chosen, it should not be of p_set

        n_set = []
        self.n_set_chosen_identities_indices = []

        nstar_set = []
        nstar_set_chosen_identities_indices = []

        for identity_id, available_instance_indices_of_this_identity in n_set_identities.items():

            # n_set:
            indices = deepcopy(available_instance_indices_of_this_identity)
            for instance in range(self.n_set_instances_per_identity - self.nstar_set_instances_per_identity):
                # second of the two places to insert randomness:
                chosen_index = random.choice(indices)
                self.n_set_chosen_identities_indices.append(chosen_index)
                n_set.append(self.X_all_faces[ chosen_index ])
                indices.remove(chosen_index)

            # nstar_set:
            for instance in range(self.nstar_set_instances_per_identity):
                # second of the two places to insert randomness:
                chosen_index = random.choice(indices)
                nstar_set_chosen_identities_indices.append(chosen_index)
                nstar_set.append(self.X_all_faces[ chosen_index ])
                indices.remove(chosen_index)

        n_set = np.array(n_set)
        nstar_set = np.array(nstar_set)




        try:
            p_set_clusters = self.return_clusters(p_set, p_set_number_of_clusters)
        except: #if p_set is empty, the code above wont work, and then create the following:
            p_set_clusters = np.array([[]])

        return p_set, pstar_set, p_set_clusters, n_set, nstar_set

    def select_I_identities_each_of_which_has_S_instances(self,
                                                          number_of_identities,
                                                          instances_per_identity,
                                                          forbidden_identites_list):

        '''
        this function is a helper function of create_Pset_Nset and shouldnt be accessed anywhere else because of the seed setting.
        :return: shipped_identities --> this is a dictionary of identity lists that have at
        least instances_per_identity instances.
        It is like the following:
        { identity1: [identity1 instance1 address, identity1 instance2 address, ...], identity2: [identity2 instance1 address, identity2 instance2 address, ...] ...}
        '''

        permitted_identities = list()
        for label in self.all_faces_labels[:self.number_of_data_for_dev_purpose_only]:
            if label not in forbidden_identites_list:
                if label not in permitted_identities:
                    permitted_identities.append(label)


        #first of the two places to insert randomness:
        random.shuffle(permitted_identities)


        shipped_identities = dict()
        for permitted_identity in permitted_identities:
            if len(shipped_identities) == number_of_identities:
                break

            available_instance_indices_of_this_identity = [index for index, identity_id in enumerate(self.all_faces_labels) if identity_id == permitted_identity ]
            if len(available_instance_indices_of_this_identity) >= instances_per_identity:
                shipped_identities[permitted_identity] = available_instance_indices_of_this_identity


        if len(shipped_identities) != number_of_identities:
            raise 'could not extract this number of identities'

        return shipped_identities



    def GA_initialization(
        self,
        p_cluster,
        n_set):

        fitess_object = Weighted_PSet_NSet_LockAndDistance(
            p_data = p_cluster,
            p_data_lock_weight= self.p_data_lock_weight,
            n_data = n_set,
            n_data_lock_weight = self.n_data_lock_weight,
            p_n_balance_weight = self.p_n_balance_weight,
            threshold_for_p = self.p_set_threshold,
            threshold_for_n = self.n_set_threshold,
            dist_function = self.distance_func)


        my_GA = Lmmaes(
            fit_object=fitess_object,
            number_of_generations = self.GA_number_of_generations,
            sigma=self.GA_sigma,  # Initial step size
            popsize = self.GA_population_size,  # Custom population size
            rseed=1)
        return my_GA




    @staticmethod
    def clip_to_0_255_and_change_dtype_to_uint8(img):
        return np.clip(img, a_min=0, a_max=255).astype(np.uint8)

    @staticmethod
    def peal_off_a_square_patch_from_center(img, patch_size):
        height, width, _ = img.shape
        height_center = int(height / 2)
        width_center = int(width / 2)

        patch_half_size = int(patch_size / 2)

        img[height_center - patch_half_size: height_center + patch_half_size,
        width_center - patch_half_size: width_center + patch_half_size] = 0.

        return img

    def preprocess_autoencoder_face(self, ae_a_face):
        '''
        deprocated; use preprocess_autoencoder_face_mtcnn_fit instead
        '''

        ae_a_face = ae_a_face[:,:,:3]
        if np.max(ae_a_face) < 2 : # if [0,1], convert to [0,255]
            ae_a_face *= 255
            ae_a_face = self.clip_to_0_255_and_change_dtype_to_uint8(ae_a_face)

        ae_a_face_resized = img_resize_pil(ae_a_face, self.img_size)
        ae_a_face_resized_normalized = self.img_normalizer_function(ae_a_face_resized)
        return ae_a_face_resized_normalized.reshape(1,self.img_size, self.img_size,3)


    def preprocess_autoencoder_face_mtcnn_fit(self, ae_face):
        ae_face = ae_face[:,:,:3]
        if np.max(ae_face) < 2 : # if [0,1], convert to [0,255]
            ae_face *= 255
            ae_face = self.clip_to_0_255_and_change_dtype_to_uint8(ae_face)

        ae_face = self.make_image_fit_to_mtcnn(ae_face)

        ae_face = self.img_normalizer_function(ae_face)

        return ae_face.reshape(1,self.img_size, self.img_size,3)

    def make_image_fit_to_mtcnn(self, img):
        '''
        makes the input image fit completely for mtcnn, so that if mtcnn is called on output image,
        the same exact image will be returned.
        for example if the input image is 160 by 160, the output of mtcnn is usually less than it according to
        the humans face position. this method makes the image to be output of mtcnn of exact size of 160 by 160
        '''

        img = self.face_detection(img)

        while img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = img_resize_pil(img, self.img_size)
            img = np.asarray(img, 'uint8')
            img = self.face_detection(img)

        return img

    def loss_log(self, iter, loss, file_name):

        with open(self.dir_loss_history + '/'+ file_name, 'a') as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerow([iter, loss])
            f_object.close()
    def save_imgs(self,
                  cluster_index,
                  method_name,
                  generator_input,
                  formAvg_formAvgPatch_or_ae_deriveImg,
                  cluster_count,
                  seed):

        if formAvg_formAvgPatch_or_ae_deriveImg == 'fa':
            gen_imgs = self.form_average_generator.predict(generator_input)
        if formAvg_formAvgPatch_or_ae_deriveImg == 'ae':
            gen_imgs = self.ae_generator.predict(generator_input)
        if formAvg_formAvgPatch_or_ae_deriveImg == 'deriveImg':
            gen_imgs = generator_input

        gen_imgs = self.img_unnormalizer_function(gen_imgs)

        gen_imgs = self.clip_to_0_255_and_change_dtype_to_uint8(gen_imgs)

        imageB = gen_imgs[0].reshape(self.img_size, self.img_size, 3)
        imageA = self.clip_to_0_255_and_change_dtype_to_uint8(
            self.img_unnormalizer_function( self.deriveImg_a_face).reshape(self.img_size, self.img_size, 3))
        # Compute SSIM for each channel independently
        ssim_r = compare_ssim(imageA[:, :, 0], imageB[:, :, 0])
        ssim_g = compare_ssim(imageA[:, :, 1], imageB[:, :, 1])
        ssim_b = compare_ssim(imageA[:, :, 2], imageB[:, :, 2])

        # Combine the SSIM measures
        ssim = (ssim_r + ssim_g + ssim_b) / 3

        print("SSIM: ", ssim)

        plt.imshow(gen_imgs[0].reshape(self.img_size, self.img_size, 3))


        if formAvg_formAvgPatch_or_ae_deriveImg == 'fa':
            dir_ = self.form_average_img_dir + '/' + method_name + str(cluster_count) +'_clstrs_'+ str(seed)+'seed'
            if not os.path.exists(dir_): os.makedirs(dir_)
            plt.savefig(dir_+'/cluster_%06d.png' % cluster_index, bbox_inches='tight', pad_inches=0)
        if formAvg_formAvgPatch_or_ae_deriveImg == 'ae':
            dir_ = self.auto_encoder_img_dir + '/' + method_name+ str(cluster_count) +'_clstrs_'+ str(seed)+'seed'
            if not os.path.exists(dir_): os.makedirs(dir_)
            plt.savefig(dir_+'/cluster_%06d.png' % cluster_index, bbox_inches='tight', pad_inches=0)
        if formAvg_formAvgPatch_or_ae_deriveImg == 'deriveImg':
            dir_ = self.deriveImg_img_dir + '/' + method_name + str(cluster_count) +'_clstrs_'+ str(seed)+'seed'
            if not os.path.exists(dir_): os.makedirs(dir_)
            plt.savefig(dir_+'/cluster_%06d.png' % cluster_index, bbox_inches='tight', pad_inches=0)


        plt.close()
        return ssim


    def return_combined_model_for_form_average(self):
        optimizer = Adam(0.0002, 0.5)
        z = Input(shape=(self.latent_dim,))
        self.form_average_generator._name= 'generator_model' #added to prevent an error caused when face_descriptor is facenet and l2 is False
        img = self.form_average_generator(z)

        self.face_descriptor.trainable = False

        embedding = self.face_descriptor(img)

        combined = Model(z, embedding, name='form_average_combined_model') #name added to prevent an error caused when face_descriptor is facenet and l2 is False
        combined.compile(loss=self.form_average_loss_function, optimizer=optimizer)

        self.form_average_combined_model = combined



    def return_combined_model_for_ae(self):
        optimizer = Adam(0.0002, 0.5)
        original_img = Input(shape=(self.img_shape[0],self.img_shape[1],self.img_shape[2],))
        img = self.ae_generator(original_img)

        self.face_descriptor.trainable = False

        embedding = self.face_descriptor(img)

        combined = Model(inputs=original_img, outputs = [img, embedding])
        combined.compile(loss=[self.ae_loss_function_maintain_a_face, self.ae_loss_function_maintain_a_master_embedding],
                         loss_weights=self.ae_loss_weights, optimizer=optimizer)

        self.ae_combined_model = combined


    def GA_loss_report(self, my_cluster,
                       number,
                       my_sum_from_p,
                       my_not_crackeds_of_p,
                       my_sum_from_n,
                       my_not_crackeds_of_n,
                       pop_size):
        cols = ['cluster',
                'number',
                'sum of distances from p-set',
                'number of NOT cracked locks in p-set',
                'sum of distances from n-set',
                'number of NOT cracked locks in n-set',
                'population size']
        row = [
                str(my_cluster+1),
                number,
                my_sum_from_p,
                my_not_crackeds_of_p,
                my_sum_from_n,
                my_not_crackeds_of_n,
                pop_size]

        file_path = self.dir_config+'/loss_ga.csv'
        file_exists = os.path.exists(file_path)
        with open(file_path, 'a') as csv_file:
            writer = csv.writer(csv_file)
            #if a new csv file is being created:
            if file_exists == False:
                writer.writerow(cols)
            writer.writerow(row)


    def return_clusters(self, p_set, p_set_number_of_clusters):
        kmeans = KMeans(n_clusters=p_set_number_of_clusters, random_state=0).fit(p_set.reshape(-1, p_set.shape[-1]) )
        cluster_labels = kmeans.labels_
        clusters = []
        set_of_labels = np.unique(cluster_labels)
        for label in set_of_labels:
            clusters.append( p_set[label == cluster_labels] )
        return clusters


    def save_to_csv_local(self,
                          pset_unlockeds,
                          pset_sum_of_dists,
                          pset_count_of_unlockeds,
                          pstar_set_unlockeds,
                          nset_unlockeds,
                          nset_sum_of_dists,
                          nset_count_of_unlockeds,
                          nstar_set_unlockeds,
                          clstr_index,
                          clstr_count,
                          method_name,
                          seed,
                          ssim):

        self.ssim_list.append(ssim)

        pset_unlockeds_percent = self.local_coverage_calc_helper(pset_unlockeds)
        nset_unlockeds_percent = self.local_coverage_calc_helper(nset_unlockeds)

        pstar_set_unlockeds_percent = self.local_coverage_calc_helper(pstar_set_unlockeds)
        nstar_set_unlockeds_percent = self.local_coverage_calc_helper(nstar_set_unlockeds)
        cols = ['phase',
                'cluster',
                'of clusters',
                'single point coverage in p-set',
                'global coverage in p-set',
                'single point coverage in n-set',
                'global coverage in n-set',
                'p-set size',
                'n-set size',
                'seed',
                'p-set cracked locks',
                'p-set sum of distances',
                'n-set cracked locks',
                'n-set sum of distances',
                'ssim']
        row=[   method_name,
                str(clstr_index+1),
                str(clstr_count),
                pset_unlockeds_percent,
                '-',
                nset_unlockeds_percent,
                '-',
                str( len(pset_unlockeds) ),
                str( len(nset_unlockeds) ),
                str(seed),
                str(pset_count_of_unlockeds),
                str(pset_sum_of_dists),
                str(nset_count_of_unlockeds),
                str(nset_sum_of_dists),
                str(ssim)]

        file_path = self.dir_config+'/'+self.results_csv_file_name
        file_exists = os.path.exists(file_path)
        with open(file_path, 'a') as csv_file:
            writer = csv.writer(csv_file)
            #if a new csv file is being created:
            if file_exists == False:
                writer.writerow(cols)
            writer.writerow(row)


    def save_to_csv_global(self,
                           pset_unlockeds,
                           nset_unlockeds,
                           pstar_set_unlockeds,
                           nstar_set_unlockeds,
                           method_name,
                           clstr_count,
                           seed):
        if pset_unlockeds == -2:
            row = [method_name,
                   'summary',
                   '-',
                   '-',
                   '-',
                   '-',
                   '-',
                   '-',
                   '-',
                   '-',
                   '-',
                   '-',
                   '-',
                   '-',
                   '-']
        else:

            ssim_list = np.array(self.ssim_list)
            ssim_mean = np.mean(ssim_list)
            ssim_std = np.std(ssim_list)
            ssim_summary = 'mean= ' + str(ssim_mean)+ ' - std= '+ str(ssim_std)
            self.ssim_list = list()

            pset_unlockeds_percent = self.accumulative_coverage_calc_helper(pset_unlockeds)
            nset_unlockeds_percent = self.accumulative_coverage_calc_helper(nset_unlockeds)

            pstar_set_unlockeds_percent = self.accumulative_coverage_calc_helper(pstar_set_unlockeds)
            nstar_set_unlockeds_percent = self.accumulative_coverage_calc_helper(nstar_set_unlockeds)

            row=[method_name,
                    'accumulation',
                    str(clstr_count),
                    '-',
                    pset_unlockeds_percent,
                    '-',
                    nset_unlockeds_percent,
                    str( len(pset_unlockeds[0]) ),
                    str( len(nset_unlockeds[0]) ),
                    str(seed),
                    '-',
                    '-',
                    '-',
                    '-',
                    ssim_summary ]
        # ['cluster',
        #  'of clusters',
        #  'single point coverage in p-set',
        #  'global coverage in p-set',
        #  'single point coverage in n-set',
        #  'global coverage in n-set',
        #  'p-set size',
        #  'n-set size',
        #  'seed',
        #  'p-set cracked locks',
        #  'p-set sum of distances',
        #  'n-set cracked locks',
        #  'n-set sum of distances']
        file_path = self.dir_config+'/'+self.results_csv_file_name
        with open(file_path, 'a') as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(row)

    def master_tester(self, potential_master_embedding, data, datastar):
        opened = []
        opened_datastar = []

        if data.shape[-1] == 0 : #if data is empty
            return opened, -1 , -1, opened_datastar
        data = data.reshape(-1, 1, data.shape[-1])
        potential_master_embedding = potential_master_embedding.reshape(1, -1)

        for d in data:
            if self.distance_func(d, potential_master_embedding) < self.thresh_in_use:
                opened.append(1)
            else:
                opened.append(0)

        for d in datastar:
            if self.distance_func(d, potential_master_embedding) < self.thresh_in_use:
                opened_datastar.append(1)
            else:
                opened_datastar.append(0)

        my_sum_of_dists = self.sum_of_dists(potential_master_embedding=potential_master_embedding, data=data)
        my_count_of_unlockeds = self.count_of_unlockeds(potential_master_embedding=potential_master_embedding, data=data)
        return opened, my_sum_of_dists, my_count_of_unlockeds, opened_datastar

    def sum_of_dists(self, potential_master_embedding, data):
        return sum(self.distance_func(potential_master_embedding, d) for d in data)

    def count_of_unlockeds(self,potential_master_embedding, data):
        return sum(self.distance_func(potential_master_embedding, d) < self.thresh_in_use for d in data)


    def train(self):
        start_time = time.time()

        for pset_clusters in self.p_set_number_of_clusters_list:
            for pset__ in self.p_set_number_of_identities_list:
                for nset__ in self.n_set_number_of_identities_list:
                    for seed in self.seed_list:
                        p_set, pstar_set, p_set_clusters, n_set, nstar_set = self.create_Pset_Nset(
                                                p_set_number_of_clusters=pset_clusters,
                                                p_set_number_of_identities = pset__,
                                                n_set_number_of_identities = nset__,
                                                seed = seed)


                        centroids = []
                        ga_outputs = []

                        # Calculating the coverage of the centroids================================:
                        # unluckeds_for_centroids_pset = []
                        # unluckeds_for_centroids_pstar_set = []
                        # unluckeds_for_centroids_nset = []
                        # unluckeds_for_centroids_nstar_set = []
                        # for index, cluster in enumerate(p_set_clusters):
                        #
                        #     centroid = np.mean(cluster, axis=0)
                        #     if self.apply_l2 == True:
                        #         centroid = l2norm(centroid)
                        #
                        #     centroids.append(centroid)
                        #
                        #     unluckeds_for_this_cluster_pset, my_sum_of_dists_pset, my_count_of_locks_pset, unluckeds_for_this_cluster_pstar_set = \
                        #         self.master_tester(centroid, data=p_set, datastar=pstar_set)
                        #     unluckeds_for_centroids_pset.append(unluckeds_for_this_cluster_pset)
                        #     unluckeds_for_centroids_pstar_set.append(unluckeds_for_this_cluster_pstar_set)
                        #
                        #     unluckeds_for_this_cluster_nset, my_sum_of_dists_nset, my_count_of_locks_nset, unluckeds_for_this_cluster_nstar_set = \
                        #         self.master_tester(centroid, data=n_set, datastar=nstar_set)
                        #     unluckeds_for_centroids_nset.append(unluckeds_for_this_cluster_nset)
                        #     unluckeds_for_centroids_nstar_set.append(unluckeds_for_this_cluster_nstar_set)
                        #
                        #     self.save_to_csv_local(pset_unlockeds =unluckeds_for_this_cluster_pset,
                        #                            pset_sum_of_dists=my_sum_of_dists_pset,
                        #                            pset_count_of_unlockeds=my_count_of_locks_pset,
                        #                            pstar_set_unlockeds=unluckeds_for_this_cluster_pstar_set,
                        #                            nset_unlockeds=unluckeds_for_this_cluster_nset,
                        #                            nset_sum_of_dists=my_sum_of_dists_nset,
                        #                            nset_count_of_unlockeds=my_count_of_locks_nset,
                        #                            nstar_set_unlockeds=unluckeds_for_this_cluster_nstar_set,
                        #                            clstr_index=index,
                        #                            clstr_count=pset_clusters,
                        #                            method_name='centroid',
                        #                            seed = seed)
                        # # saving accumulative results:
                        # self.save_to_csv_global(pset_unlockeds=unluckeds_for_centroids_pset,
                        #                         nset_unlockeds=unluckeds_for_centroids_nset,
                        #                         pstar_set_unlockeds=unluckeds_for_centroids_pstar_set,
                        #                         nstar_set_unlockeds=unluckeds_for_centroids_nstar_set,
                        #                         method_name='centroid',
                        #                         clstr_count=pset_clusters,
                        #                         seed= seed)
                        #:Calculating the coverage of the centroids================================

                        # Calculating the coverage of the GA output:================================
                        unluckeds_for_ga_pset = []
                        unluckeds_for_ga_nset = []
                        unluckeds_for_ga_pstar_set = []
                        unluckeds_for_ga_nstar_set = []
                        for index, cluster in enumerate(p_set_clusters):

                            my_GA = self.GA_initialization(cluster, n_set)

                            for gen in range(my_GA.number_of_generations):
                                my_GA.step()
                                if gen % 100 == 0:
                                    self.GA_loss_report(my_cluster=index,
                                                        number=gen,
                                                        my_sum_from_p=my_GA.fit_object.sum_of_dist_from_p(
                                                            my_GA.best_ind),
                                                        my_not_crackeds_of_p=my_GA.fit_object.count_of_p_locks(
                                                            my_GA.best_ind),
                                                        my_sum_from_n=my_GA.fit_object.sum_of_dist_from_n(
                                                            my_GA.best_ind),
                                                        my_not_crackeds_of_n=my_GA.fit_object.count_of_n_locks(
                                                            my_GA.best_ind),
                                                        pop_size=my_GA.popsize)

                            ga_output = my_GA.best_ind.reshape(1, -1)
                            ga_outputs.append(ga_output)

                            unluckeds_for_this_cluster_pset, my_sum_of_dists_pset, my_count_of_locks_pset, unluckeds_for_this_cluster_pstar_set = \
                                self.master_tester(my_GA.best_ind, data=p_set, datastar=pstar_set)
                            unluckeds_for_ga_pset.append(unluckeds_for_this_cluster_pset)
                            unluckeds_for_ga_pstar_set.append(unluckeds_for_this_cluster_pstar_set)


                            unluckeds_for_this_cluster_nset, my_sum_of_dists_nset, my_count_of_locks_nset, unluckeds_for_this_cluster_nstar_set = \
                                self.master_tester(my_GA.best_ind, data=n_set, datastar=nstar_set)
                            unluckeds_for_ga_nset.append(unluckeds_for_this_cluster_nset)
                            unluckeds_for_ga_nstar_set.append(unluckeds_for_this_cluster_nstar_set)


                            self.save_to_csv_local(pset_unlockeds=unluckeds_for_this_cluster_pset,
                                                   pset_sum_of_dists=my_sum_of_dists_pset,
                                                   pset_count_of_unlockeds=my_count_of_locks_pset,
                                                   pstar_set_unlockeds=unluckeds_for_this_cluster_pstar_set,
                                                   nset_unlockeds=unluckeds_for_this_cluster_nset,
                                                   nset_sum_of_dists=my_sum_of_dists_nset,
                                                   nset_count_of_unlockeds=my_count_of_locks_nset,
                                                   nstar_set_unlockeds=unluckeds_for_this_cluster_nstar_set,
                                                   clstr_index=index,
                                                   clstr_count=pset_clusters,
                                                   method_name='GA',
                                                   seed=seed,
                                                   ssim=-2)
                        # saving accumulative results:
                        self.save_to_csv_global(pset_unlockeds=unluckeds_for_ga_pset,
                                                nset_unlockeds=unluckeds_for_ga_nset,
                                                pstar_set_unlockeds=unluckeds_for_ga_pstar_set,
                                                nstar_set_unlockeds=unluckeds_for_ga_nstar_set,
                                                method_name='GA',
                                                clstr_count=pset_clusters,
                                                seed=seed)
                        #:Calculating the coverage of the GA output================================

                        #changing the derive image image to the n-set member if the n-set size is one and use_the_only_nset_image_as_deriveImg is net to true.
                        if self.use_the_only_nset_image_as_deriveImg == True:
                            if len(n_set) !=1:
                                raise Exception('when use_the_only_nset_image_as_deriveImg is set to True, n-set size must be 1')
                            nset_img = self.all_faces[self.n_set_chosen_identities_indices[0]]
                            self.deriveImg_a_face = self.preprocess_autoencoder_face_mtcnn_fit(nset_img)

                        self.train_deriveImg(centroids=centroids,
                                               ga_outputs=ga_outputs,
                                                 p_set = p_set,
                                                 n_set = n_set,
                                                 pstar_set=pstar_set,
                                                 nstar_set=nstar_set,
                                                 seed = seed)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time elapsed: {elapsed_time} seconds")

        self.save_to_csv_global(pset_unlockeds=-2,
                                nset_unlockeds=-2,
                                pstar_set_unlockeds=-2,
                                nstar_set_unlockeds=-2,
                                method_name=f"Training time: {end_time - start_time} seconds",
                                clstr_count=-2,
                                seed=-2)

        self.save_to_csv_global(pset_unlockeds=-2,
                                nset_unlockeds=-2,
                                pstar_set_unlockeds=-2,
                                nstar_set_unlockeds=-2,
                                method_name=f"Total Time elapsed: {end_time - self.start_time} seconds",
                                clstr_count=--2,
                                seed=-2)



    @staticmethod
    def face_detection(img_):

        output = _mtcnn.detect_faces(img_)
        (x, y, width, height) = output[0]['box']

        # a bug in MTCNN: https://github.com/ipazc/mtcnn/issues/11
        if x < 0: x = 0
        if y < 0: y = 0

        extracted_face = img_[y:y + height,
                         x:x + width, :]
        return extracted_face
    def reconstruct_embedding_helper(self, generated_img, detect_face=False):
        # gen_img = generator.predict(generator_input)
        gen_img_unnormalized = self.img_unnormalizer_function(generated_img)

        gen_img_unnormalized = self.clip_to_0_255_and_change_dtype_to_uint8(gen_img_unnormalized)

        if detect_face:
            gen_img_unnormalized = gen_img_unnormalized.reshape(self.img_size, self.img_size, 3)
            gen_img_unnormalized = self.face_detection(gen_img_unnormalized)
            gen_img_unnormalized = img_resize_pil(gen_img_unnormalized, self.img_size)
            gen_img_unnormalized = gen_img_unnormalized.reshape(1, self.img_size, self.img_size, 3)

        gen_img_unnormalized = gen_img_unnormalized.astype('float64')
        gen_img_normalized = self.img_normalizer_function(gen_img_unnormalized)

        reconstructed = self.face_descriptor(gen_img_normalized)
        reconstructed = reconstructed.numpy().reshape(1, -1)
        return reconstructed


    def local_coverage_helper(self, embedding,
                              embedding_index,
                              clstr_count,
                              method_name,
                              p_set,
                              n_set,
                              pstar_set,
                              nstar_set,
                              seed,
                              ssim):


        unluckeds_for_this_cluster_pset, my_sum_of_dists_pset, my_count_of_locks_pset, unluckeds_for_this_cluster_pstar_set = \
            self.master_tester(embedding, data = p_set, datastar=pstar_set)

        unluckeds_for_this_cluster_nset, my_sum_of_dists_nset, my_count_of_locks_nset, unluckeds_for_this_cluster_nstar_set = \
            self.master_tester(embedding, data = n_set, datastar=nstar_set)



        self.save_to_csv_local(pset_unlockeds =unluckeds_for_this_cluster_pset,
                               pset_sum_of_dists=my_sum_of_dists_pset,
                               pset_count_of_unlockeds=my_count_of_locks_pset,
                               pstar_set_unlockeds=unluckeds_for_this_cluster_pstar_set,
                               nset_unlockeds=unluckeds_for_this_cluster_nset,
                               nset_sum_of_dists=my_sum_of_dists_nset,
                               nset_count_of_unlockeds=my_count_of_locks_nset,
                               nstar_set_unlockeds=unluckeds_for_this_cluster_nstar_set,
                               clstr_index=embedding_index,
                               clstr_count=clstr_count,
                               method_name=method_name,
                               seed = seed,
                               ssim=ssim)

        return unluckeds_for_this_cluster_pset, unluckeds_for_this_cluster_nset, unluckeds_for_this_cluster_pstar_set ,unluckeds_for_this_cluster_nstar_set



    def train_deriveImg_on_given_embeddings(self,
                                            candidate_master_embeddings,
                                            method_name,
                                            p_set,
                                            n_set,
                                            pstar_set,
                                            nstar_set,
                                            seed):
        unluckeds_pset_no_face = []
        unluckeds_nset_no_face = []
        unluckeds_pset_with_face = []
        unluckeds_nset_with_face = []

        unluckeds_pstar_set_no_face = []
        unluckeds_nstar_set_no_face = []
        unluckeds_pstar_set_with_face = []
        unluckeds_nstar_set_with_face = []

        cluster_count = len(candidate_master_embeddings)


        method_name_with_face = method_name #+ '_with_face_detection'
        method_name_no_face = method_name + '_without_face_detection'

        for index, candidate_master_embedding in enumerate(candidate_master_embeddings):


            gen_img = self.deriveImg(candidate_master_embedding)

            ssim = self.save_imgs(cluster_index=index,
                           method_name=method_name,
                           generator_input=gen_img,
                           formAvg_formAvgPatch_or_ae_deriveImg='deriveImg',
                           cluster_count=cluster_count,
                           seed=seed)


            # With Face Detection:===========================================

            try:  # if no faces are detected:
                reconstructed2 = self.reconstruct_embedding_helper(
                    generated_img=gen_img,
                    detect_face=True)

                unluckeds_for_this_cluster_pset, unluckeds_for_this_cluster_nset,\
            unluckeds_for_this_cluster_pstar_set, unluckeds_for_this_cluster_nstar_set  = self.local_coverage_helper(
                    embedding=reconstructed2,
                    embedding_index=index,
                    clstr_count=cluster_count,
                    method_name=method_name_with_face,
                    p_set=p_set,
                    n_set=n_set,
                    pstar_set=pstar_set,
                    nstar_set = nstar_set,
                    seed=seed,
                    ssim=ssim)

                unluckeds_pset_with_face.append(unluckeds_for_this_cluster_pset)
                unluckeds_nset_with_face.append(unluckeds_for_this_cluster_nset)
                unluckeds_pstar_set_with_face.append(unluckeds_for_this_cluster_pstar_set)
                unluckeds_nstar_set_with_face.append(unluckeds_for_this_cluster_nstar_set)

            except:
                self.save_to_csv_local(
                    pset_unlockeds=[0],
                    pset_sum_of_dists=-1,
                    pset_count_of_unlockeds=-1,
                    pstar_set_unlockeds=[0],
                    nset_unlockeds=[0],
                    nset_sum_of_dists=-1,
                    nset_count_of_unlockeds=-1,
                    nstar_set_unlockeds=[0],
                    clstr_index=index,
                    clstr_count=len(candidate_master_embeddings),
                    method_name=method_name_with_face,
                    seed=seed,
                    ssim=ssim)

                unluckeds_pset_with_face.append(np.zeros((p_set.shape[0],)))
                unluckeds_nset_with_face.append(np.zeros((n_set.shape[0],)))
                unluckeds_pstar_set_with_face.append(np.zeros((pstar_set.shape[0],)))
                unluckeds_nstar_set_with_face.append(np.zeros((nstar_set.shape[0],)))

            #:With Face Detection===========================================

        self.save_to_csv_global(pset_unlockeds=unluckeds_pset_with_face,
                                nset_unlockeds=unluckeds_nset_with_face,
                                pstar_set_unlockeds=unluckeds_pstar_set_with_face,
                                nstar_set_unlockeds=unluckeds_nstar_set_with_face,
                                method_name=method_name_with_face,
                                clstr_count=cluster_count,
                                seed=seed)





    def deriveImg(self,
                  destination_embedding):
        '''
        xadv = deepcopy(self.deriveImg_a_face)
        xadv = xadv.reshape(1, self.img_size, self.img_size, 3)
        xadv = tf.cast(xadv, tf.float32)

        for i in range(self.deriveImg_iterations ):
            with tf.GradientTape() as tape:
                tape.watch(xadv)
                prediction = self.face_descriptor(xadv)
                loss = self.deriveImg_loss_function(prediction, destination_embedding)
                if i % 10 == 0: print(loss)

            # Get the gradients of the loss w.r.t to the input image.
            gradient = tape.gradient(loss, xadv)
            #optionA:

            # Get the sign of the gradients to create the perturbation
            signed_grad = tf.sign(gradient)
            xadv = xadv - self.deriveImg_epsilon * signed_grad
            # xadv = xadv - eps * gradient
        '''
        optimizer = Adam()

        xadv = deepcopy(self.deriveImg_a_face)
        xadv = xadv.reshape(1, self.img_size, self.img_size, 3)

        for i in range(self.deriveImg_iterations ):
            xadv = tf.Variable(xadv)

            with tf.GradientTape() as tape:
                tape.watch(xadv)
                prediction = self.face_descriptor(xadv)
                loss = self.deriveImg_loss_function(prediction, destination_embedding)
                if i % 10 == 0:
                    self.loss_log(iter=i,
                                  loss=tf.math.reduce_sum(loss).numpy(),
                                  file_name='deriveImg_loss.csv')

            # Get the gradients of the loss w.r.t to the input image.
            gradient = tape.gradient(loss, xadv)
            #optionA:
            optimizer.apply_gradients(zip([gradient], [xadv]))
            '''
            #optionB
            # Get the sign of the gradients to create the perturbation
            signed_grad = tf.sign(gradient)
            xadv = xadv - self.deriveImg_epsilon * signed_grad
            # xadv = xadv - eps * gradient
            '''

            if self.deriveImg_perturbation_limit != None:
                perturbation = self.deriveImg_a_face - xadv
                if i % 30 == 0: print(np.max(perturbation))
                perturbation = tf.clip_by_value(perturbation, -self.deriveImg_perturbation_limit, self.deriveImg_perturbation_limit)
                xadv = self.deriveImg_a_face - perturbation

            xadv = tf.clip_by_value(xadv, self.clip_to_min, self.clip_to_max)

        return xadv




    def train_deriveImg(self, centroids, ga_outputs,
                        p_set,
                        n_set,
                        pstar_set,
                        nstar_set,
                                                 seed):
        # self.train_deriveImg_on_given_embeddings(
        #                                            candidate_master_embeddings=centroids,
        #                                            method_name='reconstructed_centroids_via_deriveImg',
        #                                            p_set = p_set,
        #                                            n_set = n_set,
        #                                             pstar_set = pstar_set,
        #                                            nstar_set = nstar_set,
        #                                           seed = seed)
        self.train_deriveImg_on_given_embeddings(
                                                   candidate_master_embeddings=ga_outputs,
                                                   method_name='reconstructed_GA_outputs_via_deriveImg',
                                                   p_set = p_set,
                                                   n_set = n_set,
                                                pstar_set=pstar_set,
                                                nstar_set=nstar_set,
                                                  seed = seed)






