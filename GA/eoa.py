from GA.prob.problems import *
from GA.opti.de import DE

from GA.opti.cmaes import CMAES
from GA.opti.cmaes_origin import CMAESO
from GA.opti.cmaes_maes import CMAESM
from GA.opti.cmaes_large import CMAESL

# beta
from GA.opti.cmaes_bipop import CMAESB

if __name__ == "__main__":
    loaded = np.load('../datasets/lfw_test_125_94_funneled_pairs_resized160_facenet_embeddings512.npz')
    data = loaded['X_embedding']

    TaskProb = LockCountAndDistanceSum(data=data,

                           threshold=0.98)
    Task = CMAESM(TaskProb, 50000)
    Task.run()

    # TaskProb = Sphere(50, -2, -1)
    # Task = DE(TaskProb, 500)
    # Task.run()
