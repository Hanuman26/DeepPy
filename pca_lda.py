import numpy as np
from  sklearn.datasets import load_digits
import matplotlib.pyplot as  plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.lda import LDA
import matplotlib.cm as cm

digits = load_digits()
data = digits.data

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

pca = PCA(n_components=10)
data_r = pca.fit(data).transform(data)

print('explained variance ratio (first two components) : %s' %str(pca.explained_variance_ratio_))

print('sum of explained variance (first two components) : %s' %str(sum(pca.explained_variance_ratio_)))

x = np.arange(10)
ys = [i+x+(i*x)**2 for i in range(10)]

plt.figure()

colors = cm.rainbow(np.linspace(0,1,len(ys)))
for c,i,target_name in zip(colors, [1,2,3,4,5,6,7,8,9,10],labels):
    plt.scatter(data_r[labels == i, 0], data_r[labels == i, 1],c=c,alpha = 0.4)
    plt.legend()
    plt.title('Scatter plot of Points plotted in first\n'
            '10 Principal Components')
plt.show()




