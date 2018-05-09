import numpy as np
import os
import math as mt
import random
import cv2
import copy

from utils      import *
from MyNets     import MyNet
from sklearn.model_selection    import train_test_split
from keras.models               import load_model, Model
from ClusteringModels           import *
from tsne  import tsneExample

if os.name == 'nt':
    from utils_image import *


################################################################################
################################################################################
def loadCenters(centers_file, w ,h):
    dataset = []
    cont    = 1
    final   = len(centers_file)
    for file in centers_file:
        #print(file)
        if os.name == 'posix':
            file = file.replace( 'y:', '/rensso/qnap')
        u_progress(cont, final)
        centers = [[],[]]
        for line in open(file.strip(), 'r'):
            item = line.split(' ')
            centers[0].append(float(item[0]))
            centers[1].append(float(item[1]))
        
        #normalizing to 0 - 1
        centers[0]  = np.divide(centers[0], w)
        centers[1]  = np.divide(centers[1], h)
        #extracting the mass center
        #m0, m1      = np.mean(centers, axis = 1)
        #centers[0]  = np.subtract(centers[0], m0) 
        #centers[1]  = np.subtract(centers[1], m1)
        dataset.append(np.array(centers))
        cont += 1

    shape       = dataset[0].shape
    dataset     = np.reshape(dataset, (len(dataset), shape[0] , shape[1])) 
    return dataset, shape

################################################################################
################################################################################
def loadCentersVector(file_dir, directory, test_size, frm_w, frm_h):
    file_list       = u_loadFileManager(file_dir)
    dataset, shape  = loadCenters(file_list, frm_w, frm_h)

    x_train, x_test, y_train, y_test, idx, idy = train_test_split(
            dataset, dataset, file_list, test_size=test_size, random_state=1)

    u_saveList2File (directory + '/train_net.lst', idx)
    u_saveList2File (directory + '/test_net.lst', idy)

    return x_train, x_test, y_train, y_test, shape

################################################################################
################################################################################
def loadCentersVectorSingle2(file_lst, frm_w, frm_h):
    
    file_list       = u_loadFileManager(file_lst)
    lst, shape      = loadCenters(file_list, frm_w, frm_h)

    return lst, shape

################################################################################
################################################################################
def create_dataset(dataset, stride, look_back=1):
    data = []
    for item in dataset:
        for i in range(0, len(item[0])-look_back+1, stride):

            x = item[0][i:(i+look_back)]
            y = item[1][i:(i+look_back)]
         
            data.append([x,y])

    return np.array(data)

################################################################################
################################################################################
def re_an_train(fil, data_info):
    directory   = data_info['directory']
    frame_w     = data_info['frame_w']
    frame_h     = data_info['frame_h']
    net_info    = data_info['net']
    feat_info   = data_info['feature']
    
    file_list   = data_info['file_list']
    test_size   = data_info['test_size']
    retr_flag   = data_info['retrain_flag']
    epochs      = data_info['epochs']

    models      = net_info['names']
    
    stride      = feat_info['stride']
    blk_size    = feat_info['blk_size']
        
    u_mkdir(directory)
    
    #..........................................................................
    #loading and preparing data
    
    if retr_flag:
        train, x_test, y_train, y_test, shape = loadCentersVector (file_list, directory, test_size, frame_w, frame_h)
    else:
        train, shape = loadCentersVectorSingle2 (directory + '/train_net.lst', frame_w, frame_h)
    
    point_size  = train[0].shape[1]

    train       = create_dataset(train, stride, blk_size)
    
    batch_size  = mt.floor ( (point_size - blk_size ) / stride ) + 1
    len_obs     = train.shape[2]

    folder      = directory + '/net_models'
    u_mkdir(folder)
    
    model_info = { 
                    'batch_size'    : batch_size,
                    'len_obs'       : len_obs,
                    'stride'        : stride,
                    'blk_size'      : blk_size,
                    'frame_w'       : frame_w,
                    'frame_h'       : frame_h
                 }

    for model in models:
        net = MyNet(model, int(batch_size), int(len_obs), 2)
        model_name = folder + '/' + net.type_ + '_'+ str(batch_size) + '.h5'
        net.train(train, epochs, model_name)
        print(model_name)

        model_info['model_name'] = model_name
        out_name = folder + '/' + net.type_ + '_'+ str(batch_size) + '.info'
        
        print('Saving model info in:' , out_name)
        with open(out_name, 'w') as fp:
            json.dump(model_info, fp)
        
        out_name = folder + '/' + net.type_ + '_'+ str(batch_size) + '.conf'
        print('Saving configuration info in:' , out_name)
        with open(out_name, 'w') as fp:
            json.dump(data_info, fp)
        
        
    
################################################################################
################################################################################
def featExtraction(fil, data_info):
    
    
    file_list   = data_info['file_list']
    models_file = data_info['models_file']
    directory   = data_info['directory']
    frame_w     = data_info['frame_w']
    frame_h     = data_info['frame_h']
    out_token   = data_info['out_token']

    #...........................................................................
    #file_list  = file_list.replace('/rensso/qnap', 'y:')


    main_test, shape    = loadCentersVectorSingle2(file_list, frame_w, frame_h) 
    point_size          = main_test[0].shape[1]

    folder = directory + '/features'
    u_mkdir(folder)

    for model_info in models_file:

        info        = json.load(open ( directory + '/net_models/' + model_info + '.info' ))
        
        model_name  = info['model_name']
        #model_name  = model_name.replace('/rensso/qnap', 'y:')
        stride      = info['stride']
        blk_size    = info['blk_size']
        
        batch_size  = info['batch_size']
        len_obs     = info['len_obs']

        #...........................................................................
        
        test    = create_dataset(main_test, stride, blk_size)         

        net     = load_model(model_name)
        
        net.summary()

        imodel      = Model(inputs  = net.input,
                            outputs = net.get_layer('feat').output)

        outs = []
        for i in range(0, len(test), batch_size):

            vec = np.array(test[i:i+batch_size])
            out = imodel.predict(vec)

            #out = net.predict(vec)
            #sub = np.subtract(vec, out)
            #out = np.linalg.norm(sub)
            
            out = out.reshape(1, out.shape[0] * out.shape[1])
            outs.append(out)

        outs = np.array(outs)
        outs = outs.reshape(outs.shape[0], outs.shape[2])
        print(outs.shape)

        feat_file   = folder + '/' + model_info + '_'+ out_token +'.ft'
        np.savetxt( feat_file, outs,
                    header   = file_list )
        
        
################################################################################
################################################################################
def clustering(file, data_info):
    
    directory   = data_info['directory']
    methods     = data_info['methods']
    args        = data_info['args']
    models      = data_info['models']
    token       = data_info['token']

    # creating directory........................................................
    folder      = directory + '/clusters'
    u_mkdir(folder)

    # loading labels............................................................ 
    labels      = u_fileList2array(data_info['labels']) 

    # clustering 
    '''
    Affinity does not need the number of clusters
    DBSCAN has fixed eps 
    SOM has fixed neurons
    '''
    funcs   = { "KMeans"    : clusteringKmeans,
                "GMM"       : clusteringGMM,
                "Affinity"  : clusteringAffinityProp,
                "Spectral"  : clusteringSpectral,
                "DBScan"    : clusteringDBScan,
                "SOM"       : clusteringSOM,
                "Birch"     : clusteringBirch,
                "KDE"       : clusteringKde
              } 
    for model in models:
        sub_folder  = folder + '/' + model
        feat_file   = directory + '/features/' + model + '_' + token +'.ft'
        features    = np.loadtxt (feat_file)
        for method in methods:
            funcs[method](features, labels, args, sub_folder)

################################################################################
################################################################################
def visualizeClusters(file, data):
    directory       = data['directory']
    dir_name        = data['sub_dir']
    nsamples        = data['nsamples']
    width           = data['width']
    height          = data['height']
    img_size        = data['img_size']
    nsamples        = data['nsamples']      # number of samples to show
    m_n_clusters    = data['max']           # number of maximum clusters to show
    reverse_flag    = data['reverse']       # 1 increase sort 0 decrease sort
    save_flag       = data['savefig']       # 1 save figs to png 

    #...........................................................................
    xfactor = width/img_size
    yfactor = height/img_size

    for sub_directory, dirs, files in os.walk(directory): 
        if sub_directory.find(dir_name) >= 0:
            if os.path.isfile(sub_directory + '/data.json' ):

                info    = json.load(open ( sub_directory + '/data.json' ))
                cluster_files   = info['cluster_list_file']
                cluster_lens    = info['cluster_lens']

                seq = sorted(range(len(cluster_lens)), key=lambda k: cluster_lens[k])
                if reverse_flag:
                    seq = seq[::-1]

                cluster_files = u_replaceStrList(cluster_files, '/rensso/qnap', 'y:')
                
                cluster_files = u_fileList2array(cluster_files)

                if save_flag:
                    sub_dir_nm  = os.path.basename(sub_directory)
                    out_dir_figs  = sub_directory + '/'+ sub_dir_nm 
                    u_mkdir(out_dir_figs)

                #...........................................................................
                for pos in range( min( len(seq), m_n_clusters) ):
                    
                    flist   = cluster_files[seq[pos]] 
                    random.seed()

                    flist   = u_replaceStrList(flist, '/rensso/qnap', 'y:')
                    
                    
                    clist   = u_fileList2array(flist)
                    
                    

                    cdlist  = random.sample( list( range( len( clist ) ) ) , 
                                            min(nsamples, len(clist) ))
                    imgs        = []
                    flist_cd    = []
                    for cd in cdlist:
                        ifile   = clist[cd].replace('.trk', '_d3i.png') 
                        ifile = ifile.replace('/rensso/qnap', 'y:')
                        tfile   = clist[cd]
                        flist_cd.append(tfile)
                        
                        tfile = tfile.replace('/rensso/qnap', 'y:')

                        f       = open(tfile)
                        point   = f.readline().split(',')[1]
                        x , y   = point.split(' ')
                        
                        img     = cv2.imread(ifile, cv2.COLOR_BGR2RGB)
                        cv2.circle(img, 
                                   (int(float(x)/xfactor), int(float(y)/yfactor)), 
                                   radius = 6, 
                                   color=(0,255,0), 
                                   thickness=3, 
                                   lineType=8, 
                                   shift=0)
                        imgs.append(img)


                    if save_flag == 0:
                        pass
                        plotImages(imgs, 'Number of items: '+ str(cluster_lens[seq[pos]]))
                    else:
                        
                        base        = out_dir_figs + '/' + sub_dir_nm + '_cluster_' + '%03d_' % (pos)  + str(seq[pos])
                        ffig_name   = base + '.png'
                        plotImages(imgs, 'Number of items: '+ str(cluster_lens[seq[pos]]),
                                   ffig_name)
                        fcdlist     = base + '.lst'
                        u_saveList2File(fcdlist, flist_cd)
################################################################################
################################################################################
def knnAtomic(train_ft, test_ft, labels, thr, test_file, out_dir):
    anomalies       = []
    frames_tup      = []
    frames          = []
    pos     = 0
    
    #distances = []
    for te in test_ft:
        flag    = 1 
        for tr in train_ft:    
            dist    = np.linalg.norm(tr- te)
            #distances.append(dist)
            
            if dist < thr:
                #print(dist)
                flag = 0        
        #showHistogram(distances)

        if flag:        
            anomalies.append(labels[pos].replace('/rensso/qnap', 'y:'))
        pos += 1




    for i in range(len(anomalies)):
    
        x   = open(anomalies[i], 'r')
        x   = x.readline()
        frm = int(x.split(',')[0]) / 5
        
        frames_tup.append((frm, anomalies[i]))
        frames.append(frm)
        
    base_name =  os.path.basename(test_file).split('.')[0]
    u_saveArrayTuple2File(out_dir + '/' + base_name +'_th_'+ str(int(thr*100)) + '.txt', frames_tup)

    return frames_tup

################################################################################
################################################################################
def knn(file, data):
    train   = data["train_feat_list"]
    test    = data["test_feat_list"]
    thr     = data["threshold"]
    labels  = data['label_list']
    out_dir = data['directory']   

    out_dir = out_dir + '/outputs'
    u_mkdir(out_dir)

    train_ft    = np.loadtxt(train)
    test_ft     = np.loadtxt(test)
    labels      = u_fileList2array(labels)
    
    return knnAtomic(train_ft, test_ft, labels, thr, test, out_dir)
    

################################################################################
################################################################################
'''
info must be a n x 3 matrix with ini fin and type if it had 
fin end of the vector
'''
def fillVector(info, fin):
    vector  = np.zeros(fin)
    for item in info:
        vector[int(item[0]):int(item[1])] = item[2]

    return vector

################################################################################
################################################################################
def makeGtVector(file, fin):
    gt_mat  = np.loadtxt(file)
    gt      = fillVector(gt_mat, fin)
    return gt, gt_mat

################################################################################
################################################################################
def makeAnomalyVector(anomalies, fin):
    anomaly_mat = []
    for frame, file in anomalies:
        buf         = open(file)
        frame_ini   = int(buf.readline().split(',')[0])
        for line in buf:
            if len(line) > 0:
                #print(line)
                frame_fin = int ( line.split(',')[0] )
        anomaly_mat.append([frame_ini, frame_fin, 1])

    #anomaly_mat = np.array(anomaly_mat)
    anomaly_vec = fillVector(anomaly_mat, fin)

    return anomaly_vec, anomaly_mat

################################################################################
################################################################################
def validationInVector(anomalies, gt):
    size        = len(anomalies)
    #obs         = np.zeros(size)
    tp, fp      = 0, 0
    tn, fn      = 0, 0

    for i in range( size ):
        if anomalies[i] == 1 and gt[i] > 0:
            tp +=1 
        if anomalies[i] == 0 and gt[i] == 0:
            tn +=1
        if anomalies[i] == 1 and gt[i] == 0:
            fp +=1
        if anomalies[i] == 0 and gt[i] > 0:
            fn +=1

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    pre = tp / (tp + fp + 1e-10)

    return fpr, tpr, pre

################################################################################
################################################################################
def validatation(file, data):
    out_dir     = data['directory']   

    train       = data["train_feat_list"]
    test        = data["test_feat_list"]
    labels      = data['label_list']

    ini         = data["ini_th"]
    fin         = data['fin_th']
    step        = data['step']
    final_frame = data['final_frame']
    gt_file     = data['gt_file']
    range_frm   = data['range_frm']

    if "out_dir" in data:
        out_dir     = data['out_dir']
    else:
        out_dir     = out_dir + '/outputs'

    u_mkdir(out_dir)

    gt, gt_mat  = makeGtVector(gt_file, final_frame)

    train_ft    = np.loadtxt(train)
    test_ft     = np.loadtxt(test)
    labels      = u_fileList2array(labels)

    fpr, tpr, pre   = [], [] , []

    for th in np.arange(ini, fin, step):
        anomalies           = []
        miss                = []
        ans                 = knnAtomic(train_ft, test_ft, labels, th, test, out_dir)
        
        a_vec, a_mat        = makeAnomalyVector(ans, final_frame)

        fpr_s, tpr_s, pre_s = validationInVector(a_vec, gt)
        fpr.append(fpr_s)
        tpr.append(tpr_s)
        pre.append(pre_s)
 
    fpr = [1] + fpr + [0]
    tpr = [1] + tpr + [0]
    pre = [1] + pre + [0]

  
    #plotRocCurve([(fpr, tpr)], ['Net'])
    out = np.array([fpr, tpr, pre])
    out = np.transpose(out)
    u_saveArrayTuple2File(out_dir + '/metrics.txt', out)


################################################################################
################################################################################
def plotRoc(file, data):
    files   = data['files']
    labels  = data['labels']

    obs = []

    for fil in files:
        mat = np.loadtxt(fil)
        mat = np.transpose(mat)

        obs.append( (mat[0], mat[1]) )

    plotRocCurve(obs, labels)

################################################################################
################################################################################
def joinFeatures(file, data):
    group1      = data['group1']
    group2      = data['group2']
    out_token   = data['out_token']
    out_dir     = data['out_dir']

    feats1  = []
    feats2  = []

    for fil in group1:
        feats1.append(np.loadtxt(fil))

    for fil in group2:
        feats2.append(np.loadtxt(fil))

    for i in range (len(feats1)):
        join_mat    = np.concatenate((feats1[i],feats2[i]), axis = 1)
        name        = out_dir + '/' + out_token[i] + '.ft'
        print('Saving join matrix in: ', name)
        np.savetxt(name, join_mat)


################################################################################
################################################################################
def tsne(file, data):
    samples_files   = data['samples_files']

    #...........................................................................
    samples = []
    cont    = 0
    labels  = []
    for samples_file in samples_files:
        sample = np.loadtxt(samples_file)
        samples.append( sample )
        label  = np.ones( len(sample) ) * cont
        labels = np.concatenate((labels, label)) 
        cont  += 1
    
    samples = np.concatenate(samples)

    tsneExample(samples, labels.astype(int))



        



