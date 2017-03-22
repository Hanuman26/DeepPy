from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

np.random.seed()
digits = load_digits()
data = scale(digits.data)

n_sample, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples: %d, \t n_features %d" %(n_digits, n_samples,n_features))
print(79 * '_')
print('% 9s' % 'init''      time inertia homo compl v-meas ARI AMI silhouette')

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s %.2fs %i %.3f %.3f %.3f %.3f'
            %(name, (time() - t0), estimator.inertia_,
                metrics.homogeneity_score(labels, estimator.labels_),
                metrics.completeness_score(labels, estimator.labels_),
                metrics.v_measure_score(labels, estimator.labels_),
                metrics.adjusted_rand_score(data,estimator.labels_),
                metrics.silhouette_score(data, estimator.labels_,
                    metric='euclidean',
                    sample_size=sample_size)))
