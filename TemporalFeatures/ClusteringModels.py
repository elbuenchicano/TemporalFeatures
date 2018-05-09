import numpy as np

import os
import json
import shutil

from sklearn.cluster    import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth, SpectralClustering, DBSCAN, Birch
from utils              import *
from sklearn.mixture    import GaussianMixture
from minisom            import MiniSom

from sklearn.neighbors.kde import KernelDensity
from scipy.stats        import norm
from sklearn.externals  import joblib


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

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class KDECluster:
    '''
    points is a vector of vectors [[],[]]
    '''
    def __init__(self, points, bw, labels):
        if len(points) < 5: 
            self.kde_   = KernelDensity(kernel='gaussian', bandwidth=bw)
        else:
            self.kde_   = KernelDensity(kernel='epanechnikov', algorithm='ball_tree', bandwidth=bw, leaf_size = 50)
        
        self.points_    = points
        self.labels_    = labels

        self.kde_.fit(points)

    #..........................................................................
    def compare(self, cluster):
        scores_self = np.exp(self.kde_.score_samples(cluster.points_))
        scores_clus = np.exp(cluster.kde_.score_samples(self.points_))

        m_self      = max(scores_self)
        m_clus      = max(scores_clus)

        return max(m_clus, m_self)
            

################################################################################
################################################################################
def cgrow(x, base):
    ''' Matlab code sample 
    m   = 1;
    x   = 0:0.01:m;
    %ld  = 0.1448; 
    ld  = 0.0869; 
    c   = 0.7; %1.5
    d   = 0.5;
    y   = exp(-d*x/ld) - (x.^2-x) / c;
    %y   = exp(-ld)*(ld.^x) ./ factorial(x);
    %y = (2 ./ (1 + exp(-2.*x)) ) - 1 + (1-0.7616);
    th = 0.3;
    y  = (1-th) * (1-y) + th    
    '''
    ld  = 0.0869 # must be fixed values extracted from log(1e-5)
    c   = 0.7 # cumulative factor
    d   = 0.5 # linear factor
    y   = np.exp(-d*x/ld) - (x**2-x) / c
    y   = (1 - base) * (1 - y) + base
    return y

################################################################################
################################################################################
class KDEC:
    def __init__(self, max_iter, prob_th, bw):
        self.max_iter_  = max_iter
        self.prob_th_   = prob_th
        self.bw_        = bw

    def fit(self, samples):
        #  initializer whole clusters ..............................................
        clusters = []
        for i in range(len(samples)):
            clusters.append(KDECluster(samples[i], self.bw_, [i]))
             
        iter    = 0
        c_flag  = False
               
        # until max iter or convergence condition is satisfied .....................
        while iter < self.max_iter_ and c_flag != True:
            
            i   = iter / self.max_iter_
            th  = cgrow(i, self.prob_th_)

            c_flag  = True
             
            n_clusters      = len(clusters)
            group_flag      = np.zeros(n_clusters)

            clusters_temp   = []

            for i in range(n_clusters):
                #u_progress(i, n_clusters)
                if group_flag[i] == 0:
                    group_flag[i]    = 1
                    for j in range(n_clusters):
                        if i != j and group_flag[j] == 0:
                            score = clusters[i].compare(clusters[j]) 
                            #print(score)
                            if score > th:
                                clusters[i].points_ = np.concatenate( (clusters[i].points_, clusters[j].points_ ))
                                clusters[i].labels_ = clusters[i].labels_ + clusters[j].labels_
                                group_flag[j]       = 1
                                c_flag              = False

                    if c_flag == False:
                        clusters[i].kde_.fit(clusters[i].points_)

                    clusters_temp.append(clusters[i])
                
            clusters = clusters_temp
            iter    += 1

        self.clusters_ = clusters

        #define labels
        self.labels_ = np.zeros(len(samples))
        for i in range(len( clusters) ):
            for l in clusters[i].labels_:
                self.labels_[l] = i

################################################################################
################################################################################
def clusteringKde(feats, labels, args, directory):
    print('KDE clustering................................................')
    max_iter    = args['max_iter']
    prob_th     = args['prob_th']
    bw          = args['bandwidth']

    #...........................................................................
    kdec = KDEC(max_iter, prob_th, bw)
    samples = feats[:, np.newaxis]
    kdec.fit(samples)
    
    clusters    = [[] for i in range(len(kdec.clusters_))]

    # fitting kmeans
    for i in range( len( kdec.labels_ ) ):
        label = kdec.labels_[i]
        clusters[int(label)].append(labels[i])

    # saving clusters
    out_dir = directory + '/KDE' 
    savingData(out_dir, clusters, [[]])

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

    


