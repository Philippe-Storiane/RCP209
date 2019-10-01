# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 03:55:03 2019

@author: philippe
"""

import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float64')
X_train /= 255
X_train = X_train.reshape(( 60000, 784))
X_train_conv = X_train.reshape( ( 60000, 28, 28, 1))

X_test = X_test.astype('float64')
X_test /= 255
X_test = X_test.reshape( ( 10000, 784 ))
X_test_conv = X_test.reshape( ( 10000, 28, 28, 1))

#y_train = to_categorical( y_train )
#y_test= to_categorical( y_test )

X_tsne = TSNE( n_components = 2, perplexity = 30, init='pca').fit_transform( X_test )

x2d = X_tsne
labels = y_test


def convexHulls(points, labels):
    convex_hulls = []
    for i in range(10):
        convex_hulls.append(ConvexHull(points[labels==i,:]))
    return convex_hulls




def best_ellipses(points, labels):
    gaussians = []
    for i in range(10):
        gaussians.append(GaussianMixture(n_components=1, covariance_type='full').fit(points[labels==i, :]))
    return gaussians



def neighboring_hit(points, labels):
    k = 6
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    txs = 0.0
    txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    for i in range(len(points)):
        tx = 0.0
        for j in range(1,k+1):
            if (labels[indices[i,j]]== labels[i]):
                tx += 1
        tx /= k
        
        txsc[labels[i]] += tx
        nppts[labels[i]] += 1
        txs += tx
        
    for i in range(10):
        txsc[i] /= nppts[i]
        
    return txs / len( points)


def visualization(points2D, labels, convex_hulls, ellipses ,projname, nh):
    
    points2D_c= []
    for i in range(10):
        points2D_c.append(points2D[labels==i, :])
    cmap =cm.tab10
    plt.figure(figsize=(3.841, 7.195), dpi=100)
    plt.set_cmap(cmap)
    plt.subplots_adjust(hspace=0.4 )
    plt.subplot(311)
    plt.scatter(points2D[:,0], points2D[:,1], c=labels,  s=3,edgecolors='none', cmap=cmap, alpha=1.0)
    plt.colorbar(ticks=range(10))
    
    plt.title("2D "+projname+" - NH="+str(nh*100.0))
    vals = [ i/10.0 for i in range(10)]
    sp2 = plt.subplot(312)
    for i in range(10):
        ch = np.append(convex_hulls[i].vertices,convex_hulls[i].vertices[0])
        sp2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1], '-',label='$%i$'%i, color=cmap(vals[i]))
    plt.colorbar(ticks=range(10))
    plt.title(projname+" Convex Hulls")
    
    def plot_results(X, Y_, means, covariances, index, title, color):
        splot = plt.subplot(3, 1, 3)
        for i, (mean, covar) in enumerate(zip(means, covariances)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color, alpha = 0.2)
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.6)
            splot.add_artist(ell)
        plt.title(title)
    
    plt.subplot(313)
    for i in range(10):
        plot_results(
                points2D[labels==i, :],
                ellipses[i].predict(points2D[labels==i, :]),
                ellipses[i].means_,
                ellipses[i].covariances_,
                0,projname+" fitting ellipses",
                cmap(vals[i]))



labels = X_test
X_tsne = TSNE( n_components = 2, perplexity = 30, init='pca').fit_transform( X_test )
ellipses = best_ellipses(X_tsne, labels)
convex_hulls= convexHulls(X_tsne, labels)
nh= neighboring_hit(X_tsne, labels)
visualization(X_tsne, labels, convex_hulls, ellipses , 't-SNE', nh)

X_pca = PCA( n_components = 2).fit_transform( X_test )
ellipses = best_ellipses(X_pca, labels)
convex_hulls= convexHulls(X_pca, labels)
nh= neighboring_hit(X_pca, labels)
visualization(X_pca, labels, convex_hulls, ellipses , 'PCA', nh)


from keras.models import model_from_yaml

def loadModel(savename):
    with open(savename+".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model ",savename,".yaml loaded ")
    model.load_weights(savename+".h5")
    print("Weights ",savename,".h5 loaded ")
    return model

perceptron = loadModel( "perceptron.h5")
X_MP = perceptron.predict( X_test)
X_MP_tsne = TSNE( n_components = 2, perplexity = 30, init='pca').fit_transform( X_MP )
ellipses = best_ellipses(X_MP_tsne, labels)
convex_hulls= convexHulls(X_MP_tsne, labels)
nh= neighboring_hit(X_MP_tsne, labels)
visualization(X_MP_tsne, labels, convex_hulls, ellipses , 'MP_t-SNE', nh)


convnet = loadModel( "convnet.h5")
X_CNN = convnet.predict( X_test )
X_CNN_tsne = TSNE( n_components = 2, perplexity = 30, init='pca').fit_transform( X_CNN )
ellipses = best_ellipses(X_CNN_tsne, labels)
convex_hulls= convexHulls(X_CNN_tsne, labels)
nh= neighboring_hit(X_CNN_tsne, labels)
visualization(X_CNN_tsne, labels, convex_hulls, ellipses , 'CNN_t-SNE', nh)







