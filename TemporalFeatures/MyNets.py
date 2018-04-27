import numpy as np
import random


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, GRU
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

from utils import u_mkdir


#Callback to print learning rate decaying over the epoches
class LearningRatePrinter(Callback):
    def init(self):
        super(LearningRatePrinter, self).init()

    def on_epoch_begin(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print ("learning rate -> " + str(lr))

################################################################################
################################################################################
'''
ID = 'Auto_Recu' 
Return a recurrent autoencoder model
GRU 
'''
def getModelAutoencoderRecurrent(bathc_size, len_obs, dimension):

    encoder_input   = Input(batch_shape=(bathc_size, dimension, len_obs), name='main_input')

    encoder         = GRU(bathc_size, activation='sigmoid', recurrent_activation='hard_sigmoid', use_bias=True, stateful = False, name = 'feat')(encoder_input)
    
    decoder_dense   = Dense(len_obs*dimension, activation='sigmoid')(encoder)
      
    reshape_data    = Reshape((dimension, len_obs), name = 'output')(decoder_dense)

    model = Model(encoder_input, reshape_data)
    
    return model

#..............................................................................
def getModelAutoencoderRecurrent2(bathc_size, len_obs, dimension):

    encoder_input   = Input(batch_shape=(bathc_size, dimension, len_obs), name='main_input')

    encoder         = GRU(bathc_size, activation='sigmoid', recurrent_activation='hard_sigmoid', use_bias=True, stateful = False, name = 'feat')(encoder_input)

    #encoder         = Flatten()(encoder)
    
    decoder_dense   = Dense(len_obs*dimension, activation='sigmoid')(encoder)
      
    reshape_data    = Reshape((dimension, len_obs), name = 'output')(decoder_dense)

    model = Model(encoder_input, reshape_data)
    
    return model

################################################################################
################################################################################
#Main class 
class MyNet(object):
    def __init__(this, type, batch_size, len_obs, dimension):
        models          = { 'Auto_Recu' : getModelAutoencoderRecurrent,
                            'Auto_Recu2': getModelAutoencoderRecurrent2}
        this.model_     = models[type] (batch_size, len_obs, dimension)
        this.type_      = type 
        this.batch_size_= batch_size
  
    def train(this, train, test, model_name, batch_size_train, epochs, directory):
        pass

    ############################################################################
    def getLayerFeats(this, x_test, model_name, layer_name = 'feat'):
        this.model_.load_weights(model_name)
        imodel      = Model(inputs  = this.model.input,
                            outputs = this.model.get_layer(layer_name).output)
       
        return imodel.predict(x_test)
      
    ############################################################################
    def train(this, train, epochs, out_name):
        
        #Save weights
        #weights_save = ModelCheckpoint(filepath             = 'out', 
        #                               monitor              = 'val_loss', 
        #                               verbose              = 0, 
        #                               save_best_only       = True, 
        #                               save_weights_only    = False, 
        #                               mode                 = 'min')
        
        #model summary
        this.model_.summary()

        #compile model
        this.model_.compile(loss='mean_squared_error', optimizer='adadelta')

        #train
        this.model_.fit(train, train, 
                  batch_size    = this.batch_size_, 
                  epochs        = epochs,
                  verbose       = 1, 
                  shuffle       = False,
                  callbacks=[TensorBoard(log_dir='/netlog')])

        this.model_.save(out_name)
 
    def loadModel(this, file):
        this.model_ = loadModel(file)
#END CLASS
################################################################################
################################################################################


