import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import *
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Concatenate, Lambda, Input, Conv1D, AveragePooling1D, BatchNormalization
import keras.backend as K
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#dataset ongoing 
#filename = "data.nc"
#dataset = xarray.open_dataset(filename)

def barlow_loss(plambd=5e-3):
    """
    The main loss function (inner_cca_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """
    def off_diagonal(x):
        zero_diag = tf.zeros(tf.shape(x)[-1])
        return tf.linalg.set_diag(x,zero_diag) 

    def normalize_repr(z):
        z_norm = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
        return z_norm

    def inner_barlow_objective(y_true, y_pred):
        lambd =  plambd #5e-3
        o1 = o2 = int(y_pred.shape[1] // 2)
        # unpack (separate) the output of networks for view 1 and view 2
        z_a = tf.transpose(y_pred[:, 0:o1])
        z_b = tf.transpose(y_pred[:, o1:o1 + o2])
        s = tf.cast(tf.shape(z_a)[0], z_a.dtype)
        z_a_norm = normalize_repr(z_a)
        z_b_norm = normalize_repr(z_b)
        # Cross-correlation matrix.
        c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / s
        # Loss.
        on_diag = tf.linalg.diag_part(c) + (-1)
        on_diag = tf.reduce_sum(tf.pow(on_diag, 2))
        off_diag = off_diagonal(c)
        off_diag = tf.reduce_sum(tf.pow(off_diag, 2))
        loss = on_diag + (lambd * off_diag)
        return loss
    return inner_barlow_objective

def BTCCA_model(input_shape=4096,nb_layers=3):
    Xinput = Input(shape=(input_shape,1))
    Yinput = Input(shape=(input_shape,1))
    model = Sequential()
    for i in range(nb_layers):
        model.add(Conv1D(filters=4,kernel_size=8,strides=1,activation="selu",padding="same"))
    model.add(Flatten())
    encoded_l = model(Xinput)
    encoded_r = model(Yinput)
    sub_layer = Concatenate(name="encoded")([encoded_l,encoded_r])
    btcca_model = Model([Xinput,Yinput],sub_layer)
    return btcca_model

def analyse_corr_result(traces,nb_neurons):
    corr = []
    for t in traces:
        corr.append(np.corrcoef(t[:nb_neurons],t[nb_neurons:])[0][1])
    return np.array(corr).reshape(-1, 1)

def kmeans_cluster(traces,labels):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(traces)
    return np.count_nonzero(kmeans.labels_==labels)/len(traces)*100, kmeans.labels_

class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, list_examples, batch_size=100, dim=50000,n_classes=2, shuffle=True, normalize=None):
        # Constructor of the data generator.
        self.dim = dim
        self.batch_size = batch_size
        self.list_examples = list_examples
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.normalize = normalize
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.list_examples) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_examples[k] for k in indexes]
        X1,X2,y = self.__data_generation(list_IDs_temp)
        return [X1,X2],y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_examples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X1 = np.empty([self.batch_size, self.dim])
        X2 = np.empty([self.batch_size, self.dim])
        y = np.ones(self.batch_size)
        # Generate data.
        for i, ID in enumerate(list_IDs_temp):
            # Load sample
            X1[i,:] = dataset["aligned_blocks"].sel(block_index=ID[0][0]).values[:200000][::4]  #multiply trace
            X2[i,:] = dataset["aligned_blocks"].sel(block_index=ID[0][1]).values[:200000][::4]  #square trace
        return X1,X2,y

if __name__ == "__main__":
    LAYERS = 6
    EPOCHS = 5
    LAMBD = 5e-3
    print("LAYERS:",LAYERS)
    print("EPOCHS:",EPOCHS)
    print("LAMBDA:",LAMBD)
    learning_rate = 0.0001
    input_shape = 50000
    nb_train_traces = 10000
    nb_test_traces = 2000
    nb_train_models = 5
    
    input_pairs, true_labels = pickle.load(open('ID_pairs_traces_and_labels.pkl','rb'))  
    #to make the datagenrator simpler
    #input_pairs: contains the index list of pairs of successive multiply/square operation [(1,2),(3,4),...]
    #true_labels: list of collision labels for each pair [0,1,...]

    # Parameters
    train_params = {'dim': input_shape,
            'batch_size': 100,
            'n_classes': 2,
            'shuffle': True}

    test_params = {'dim': input_shape,
            'batch_size': 100,
            'n_classes': 2,
            'shuffle': False}

    partition = {}
    #get rand train/test data 
    rand_index = random.sample(range(0,input_pairs.shape[0]),input_pairs.shape[0])
    train_examples = [(input_pairs[i], true_labels[i]) for i in rand_index[:nb_train_traces]]
    test_examples = [(input_pairs[i], true_labels[i]) for i in rand_index[nb_train_traces:nb_train_traces+nb_test_traces]]
    partition['train'] = train_examples
    partition['test'] = test_examples
    # Define the generators
    training_generator = DataGenerator(partition['train'], **train_params)
    testing_generator = DataGenerator(partition['test'],**test_params)
    test_labels = [true_labels[i] for i in rand_index[nb_train_traces:nb_train_traces+nb_test_traces]]

    l_acc,l_corr,l_silhouette_score,l_pred = [], [], [],[]
    with tf.device('/device:GPU:0'):
        for m in range(nb_train_models):
            print("train model:",m)
            model_btcca = BTCCA_model(input_shape=input_shape, nb_layers=LAYERS)
            optimizer = RMSprop(learning_rate=learning_rate)
            model_btcca.compile(optimizer=optimizer,loss=[barlow_loss(plambd=LAMBD)])
            model_btcca.fit(training_generator,epochs=EPOCHS,verbose=1) 
            
            new_traces = model_btcca.predict(testing_generator)
            latent_dim = int(new_traces.shape[1] // 2)
            corr = analyse_corr_result(new_traces,latent_dim)
            res,pred = kmeans_cluster(corr,test_labels)

            l_acc.append(res)
            l_corr.append(corr)
            l_pred.append(pred)
            l_silhouette_score.append(silhouette_score(corr,pred))
            print(l_acc)