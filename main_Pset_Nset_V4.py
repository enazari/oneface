from PIL import Image
import numpy as np

from other.Pset_Nset_V4.Pset_Nset_V4 import Pset_Nset_V4
from GA.prob import problems


import argparse
import json

import importlib


parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, help="JSON Configuration file ")
args = parser.parse_args()
f = open('configs/Pset_Nset_V4/' + vars(args)['config_name'] + '.json')
args = json.load(f)




face_descriptor_module = importlib.import_module('face_descriptors.' + args['face_descriptor'])
return_model_and_unnormalizer = getattr(face_descriptor_module, 'return_model_and_unnormalizer')
face_descriptor, face_descriptor_unnormalizer = return_model_and_unnormalizer()

return_normalizer = getattr(face_descriptor_module, 'return_normalizer')
face_descriptor_normalizer = return_normalizer()


if args['number_of_data_for_dev_purpose_only'] == -1:
    number_of_data_for_dev_purpose_only = None
else:
    number_of_data_for_dev_purpose_only = args['number_of_data_for_dev_purpose_only']

if args['GA_population_size'] == -1:
    GA_population_size = None
else:
    GA_population_size = args['GA_population_size']



loss_function_module = importlib.import_module('loss_functions.' + args['deriveImg_loss_function'])
deriveImg_loss_function = getattr(loss_function_module, 'loss')



deriveImg_a_face = np.array( Image.open('datasets/autoencoder_faces/'+args['deriveImg_a_face']) )


Masterkey1 = Pset_Nset_V4(model_name='p='+str(args['p_set_number_of_identities_list'])
                                        +'_pclsr='+str(args['p_set_number_of_clusters_list'])
                                        +'_n='+str(args['n_set_number_of_identities_list'])
                                        + '_s=' + str(args['seed_list']),
                        face_descriptor=face_descriptor,
                        img_size=args['img_size'],
                        img_unnormalizer_function=face_descriptor_unnormalizer,
                        img_normalizer_function=face_descriptor_normalizer,
                        apply_l2 = args['apply_l2'],
                        number_of_data_for_dev_purpose_only = number_of_data_for_dev_purpose_only,
                        distance_func=args['distance_func'],
                        use_threshold_for_best_accuracy_or_for_closest_to_FAR001 = args['use_threshold_for_best_accuracy_or_for_closest_to_FAR001'],
                        p_set_threshold_coefficient = args['p_set_threshold_coefficient'],
                        n_set_threshold_coefficient=args['n_set_threshold_coefficient'],

                          seed_list = args['seed_list'],

                        GA_number_of_generations = args['GA_number_of_generations'],
                        GA_sigma = args['GA_sigma'],
                        GA_population_size=GA_population_size,

                        p_set_number_of_identities_list = args['p_set_number_of_identities_list'],
                        p_set_instances_per_identity = args['p_set_instances_per_identity'],
                        pstar_set_instances_per_identity = args['pstar_set_instances_per_identity'],
                        p_set_number_of_clusters_list = args['p_set_number_of_clusters_list'],
                        p_data_lock_weight = args['p_data_lock_weight'],

                        n_set_number_of_identities_list = args['n_set_number_of_identities_list'],
                        n_set_instances_per_identity = args['n_set_instances_per_identity'],
                        nstar_set_instances_per_identity=args['nstar_set_instances_per_identity'],

                          n_data_lock_weight = args['n_data_lock_weight'],

                        p_n_balance_weight = args['p_n_balance_weight'],

                        #deriveImg params:
                        deriveImg_a_face = deriveImg_a_face,
                        use_the_only_nset_image_as_deriveImg = args['use_the_only_nset_image_as_deriveImg'],
                        deriveImg_epsilon = args['deriveImg_epsilon'],
                        deriveImg_loss_function = deriveImg_loss_function,
                        deriveImg_iterations = args['deriveImg_iterations'],
                        clip_to_min = args['clip_to_min'],
                        clip_to_max = args['clip_to_max'],
                        deriveImg_perturbation_limit=args['deriveImg_perturbation_limit']
                          )

with open(Masterkey1.dir_config+'/config.json', 'w') as f:
    json.dump(args, f)

Masterkey1.train()


