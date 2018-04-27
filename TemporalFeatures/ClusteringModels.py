import numpy as np
import os
import json
import shutil

from sklearn.cluster    import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth, SpectralClustering, DBSCAN, Birch
from utils              import *
from sklearn.mixture    import GaussianMixture
from minisom            import MiniSom

#import matplotlib.pyplot as plt

################################################################################
################################################################################
def savingData (out_dir, clusters, centroids):
    u_mkdir(out_dir)
    cluster_id              = 0
    cluster_file_list       = []
    cluster_file_list_size  = []

    for cluster in clusters:
        file_name    = out_dir + '/cluster' + str(cluster_id) + '.lst' 
        cluster_file_list.append(file_name)
        u_saveList2File(file_name, cluster)
        cluster_id  += 1
        cluster_file_list_size.append(len(cluster))

    fcluster_name = out_dir + '/clusters.lst'
    u_saveList2File(fcluster_name, cluster_file_list)

    # saving centroids
    fcentroids_name = out_dir + '/centroids.txt'
    print('Save centroids in: ', fcentroids_name)
    np.savetxt(fcentroids_name, centroids)

    # saving data
    data = {
        "cluster_list_file" : fcluster_name, 
        "centroids_file"    : fcentroids_name,
        "cluster_lens"      : cluster_file_list_size
    }

    finfo_name = out_dir + '/data.json'
    print('Saving cluster info in:' , finfo_name)
    with open(finfo_name, 'w') as fp:
        json.dump(data, fp)

################################################################################
################################################################################
def clusteringKmeans(feats, labels, args, directory):
    print('Kmeans clustering..................................................')
    n_clusters  = args['n_clusters']

    #...........................................................................
    kmeans      = KMeans(n_clusters = n_clusters, random_state = 0).fit(feats)
    
    clusters    = [[] for i in range(n_clusters)]

    # fitting kmeans
    for i in range( len( kmeans.labels_ ) ):
        label = kmeans.labels_[i]
        clusters[label].append(labels[i])

    # saving clusters
    out_dir = directory + '/KMeans_c' + str(n_clusters)
    savingData(out_dir, clusters, kmeans.cluster_centers_)
    
################################################################################
################################################################################
def clusteringGMM(feats, labels, args, directory):
    print('GMM clustering.....................................................')
    n_clusters  = args['n_clusters']

    #...........................................................................
    estimator   =   GaussianMixture(n_components=n_clusters, max_iter=500, random_state=3)

    estimator.fit(feats)    
    pred = estimator.predict(feats)

    clusters    = [[] for i in range(n_clusters)]

    # fitting gmm
    for i in range( len( pred ) ):
        label = pred[i]
        clusters[label].append(labels[i])

    # saving clusters 
    out_dir = directory + '/GMM_c' + str(n_clusters)
    savingData(out_dir, clusters, estimator.means_)

################################################################################
################################################################################
def clusteringAffinityProp(feats, labels, args, directory):
    print('Affinity clustering................................................')

    af = AffinityPropagation(damping = 0.5).fit(feats)

    clusters    = [[] for i in range(len(af.cluster_centers_indices_))]

    # fitting affinity
    for i in range( len( af.labels_ ) ):
        label = af.labels_[i]
        clusters[label].append(labels[i])

    # saving clusters 
    out_dir = directory + '/Affinity'
    savingData(out_dir, clusters, af.cluster_centers_)

################################################################################
################################################################################
def clusteringSpectral(feats, labels, args, directory):
    print('Spectral clustering................................................')
    n_clusters  = args['n_clusters']
    
    #...........................................................................
    spectral = SpectralClustering(
                n_clusters=n_clusters, eigen_solver='arpack',
                affinity="nearest_neighbors")
    
    spectral.fit(feats)

    clusters    = [[] for i in range(n_clusters)]

    # fitting affinity
    for i in range( len( spectral.labels_ ) ):
        label = spectral.labels_[i]
        clusters[label].append(labels[i])

    # saving clusters 
    out_dir = directory + '/Spectral_c' + str(n_clusters)
    savingData(out_dir, clusters, [[]] )

################################################################################
################################################################################
def clusteringDBScan(feats, labels, args, directory):
    print('DBScan clustering................................................')
    db = DBSCAN(eps=7, min_samples=10).fit(feats)

    n_clusters  = len( set(db.labels_) ) 

    clusters    = [[] for i in range(n_clusters)]

    # fitting kmeans
    for i in range( len( db.labels_ ) ):
        label = db.labels_[i]
        clusters[label].append(labels[i])

    # saving clusters
    out_dir = directory + '/DBScan'
    savingData(out_dir, clusters, [[]])

################################################################################
################################################################################
def clusteringSOM(feats, labels, args, directory):
    print('Kohonen clustering................................................')
    a = 4
    b = 4
    n_clusters  = a * b

    #...........................................................................

    som = MiniSom(a, b, feats.shape[1], sigma=.8, learning_rate=0.5)
    som.random_weights_init(feats[:100])
    som.train_random(feats, 1500)

    clusters    = [[] for i in range(n_clusters)]

    for i in range( len( feats) ):
        feat = feats[i]
        x, y = som.winner(feat)
        clusters[x * b + y].append(labels[i])
        
    # saving clusters
    out_dir = directory + '/SOM'
    savingData(out_dir, clusters, [[]])

################################################################################
################################################################################
def clusteringBirch(feats, labels, args, directory):
    print('Birch clustering................................................')
    n_clusters  = args['n_clusters']

    #...........................................................................
    birch       = Birch(n_clusters = n_clusters, threshold=10.5).fit(feats)
    
    clusters    = [[] for i in range(n_clusters)]

    # fitting kmeans
    for i in range( len( birch.labels_ ) ):
        label = birch.labels_[i]
        clusters[label].append(labels[i])

    # saving clusters
    out_dir = directory + '/Birch_c' + str(n_clusters)
    savingData(out_dir, clusters, [[]])

################################################################################
################################################################################
def defineNumberOfClusters(x):
    
    n_estimators    = np.arange(1, 15)

    clfs = [GaussianMixture(n, max_iter=1000).fit(x) for n in n_estimators]
    bics = [clf.bic(x) for clf in clfs]
    aics = [clf.aic(x) for clf in clfs]

    plt.plot(n_estimators, bics, label='BIC')
    plt.plot(n_estimators, aics, label='AIC')
    plt.legend();
    plt.show()
   
################################################################################
################################################################################
def getGMMParameters(x, n_components):
    gmm     = GaussianMixture(n_components, max_iter=1000, random_state=3)
    clf     = gmm.fit(x)
    xpdf    = np.linspace(0, 2000, 100000).reshape(-1,1)
    density = np.exp(clf.score_samples(xpdf))

    plt.hist(x, 150, normed=True, alpha=0.5)
    plt.plot(xpdf, density, '-r')
    plt.show()
    return gmm.means_

################################################################################
################################################################################
def filterTrks(lst1, lst2):
    candidates  = []
    fixed       = set()
    directory   = 'y:/gt'
    u_mkdir(directory)

    for i in range(len(lst1)):
        if lst1[i] > 20 and lst1[i] < 70:
            candidates.append(lst2[i])
        else:
            fil = os.path.dirname (lst2[i])    
            fixed.add(fil)

    #for i in candidates:
    #    fil = i.replace('mtR.png', 'd3i.png')
    #    base = os.path.basename(fil)
    #    shutil.copyfile(fil, directory+'/'+base)

    out     = 'y:/gt/listfiltered.txt'
    u_saveList2File(out, fixed)

    




#-------------------------------------------------------------------------------
################################################################################
################################################################################
# deprecated
#def clusteringMeanShift(feats, labels, args, directory):
#    print('MeanShift clustering...............................................')
#    n_clusters  = args['n_clusters']

#    #...........................................................................
#    bandwidth = estimate_bandwidth(feats, quantile=0.7, n_samples=500)

#    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#    ms.fit(feats)

#    print("number of estimated clusters : %d" % len(ms.cluster_centers_))

    


