
# Needed Imports
import os
import pandas as pd
import numpy as np
#import json
#from pathlib import Path
import tensorflow as tf
#import scipy.spatial.distance as dist
#import scipy.cluster.hierarchy as hier
import scipy.stats as stats
import metabolinks.transformations as transf
import pickle

# Python files in the repository
import gan_evaluation_metrics as gem
import linear_augmentation_functions as laf

# Import needed functions from GAN_functions
#from GAN_functions import wasserstein_loss
from GAN_functions import gradient_penalty_cwgan
from GAN_functions import critic_loss_wgan
from GAN_functions import generator_loss_wgan

# Testing different learning rates
# This script is to be executed in a slurm array

# indexing the learning rate to use

idx = 0#int(os.environ["SLURM_ARRAY_TASK_ID"])

test_nodes = ('128', '256', '512', '1024')

gen_nodes_str = test_nodes[idx]

print(f"\nINFO: Job with array task id {idx} is using {gen_nodes_str} nodes in generator.")

# Reading Data
# Reading the dataset, this dataset has two copies of the samples.The 2nd copy is equal to the 1st multiplied by a constant
# unique to each sample.
human_datamatrix_base = pd.read_excel('ST001082_AN001766_HD.xlsx')
human_datamatrix = human_datamatrix_base.iloc[:-1, :-4] # Just select the rows corresponding to the dataset
mz_list = human_datamatrix_base.iloc[:-1, -4] # Select column with list of m/z values

human_datamatrix = human_datamatrix.set_index(human_datamatrix.columns[0]) # Set index as the metabolites name
human_datamatrix = human_datamatrix.replace({0:np.nan}) # Replacing 0 values as missing values
# Select one of the two copies of samples in the dataset by removing the samples ending with '.1'
human_datamatrix = human_datamatrix[[i for i in human_datamatrix.columns if not i.endswith('.1')]]

human_datamatrix = human_datamatrix.T # Transpose the dataset
# Current sample labels. Sample will consist of 'Sample Type:No Recurrence' and 'Sample Type:Recurrence'
hd_labels = list(human_datamatrix['Factors'])
blanks = human_datamatrix[human_datamatrix['Factors'] == 'Sample Type:Blank'].iloc[:,1:] # Select blank samples
blanks = blanks.replace({np.nan:0}) 
blanks = blanks.astype(float) # Get blank samples with floats (needed since datamatrix has strings)
blanks_average = blanks.mean() # Average of the blanks
# Selecting the samples belonging to either the 'No Recurrence' or 'Recurrence' 
selection = []
for i in human_datamatrix.loc[:, 'Factors']:
    if i in ['Sample Type:No Recurrence', 'Sample Type:Recurrence']:
        selection.append(True)
    else:
        selection.append(False)
human_datamatrix = human_datamatrix[selection]
# Creating the list of 'targets' (labels of samples) of the dataset with the 'No Recurrence' and 'Recurrence' classes 
hd_labels = []
for i in list(human_datamatrix.iloc[:,0]):
    if i == 'Sample Type:No Recurrence':
        hd_labels.append('No Recurrence')
    else:
        hd_labels.append('Recurrence')

human_datamatrix = human_datamatrix.iloc[:,1:]
human_datamatrix = human_datamatrix.astype(float) # Passing the values from strings to floats.
human_datamatrix = human_datamatrix.replace({np.nan:0}) - blanks_average
human_datamatrix[human_datamatrix<0] = 0
human_datamatrix = human_datamatrix.replace({0:np.nan})

# Atomic masses - https://ciaaw.org/atomic-masses.htm
#Isotopic abundances-https://ciaaw.org/isotopic-abundances.htm/https://www.degruyter.com/view/journals/pac/88/3/article-p293.xml
# Isotopic abundances from Pure Appl. Chem. 2016; 88(3): 293–306,
# Isotopic compositions of the elements 2013 (IUPAC Technical Report), doi: 10.1515/pac-2015-0503

chemdict = {'H':(1.0078250322, 0.999844),
            'C':(12.000000000, 0.988922),
            'N':(14.003074004, 0.996337),
            'O':(15.994914619, 0.9976206),
            'Na':(22.98976928, 1.0),
            'P':(30.973761998, 1.0),
            'S':(31.972071174, 0.9504074),
            'Cl':(34.9688527, 0.757647),
            'F':(18.998403163, 1.0),
            'C13':(13.003354835, 0.011078) # Carbon 13 isotope
           } 

# electron mass from NIST http://physics.nist.gov/cgi-bin/cuu/Value?meu|search_for=electron+mass
electron_mass = 0.000548579909065
mz_list = mz_list[1:] - chemdict['H'][0] + electron_mass
counts = human_datamatrix.count(axis=0)
final_mz_list = list(mz_list[list(counts >= 2)])
final_mz_list = set(final_mz_list)
human_datamatrix = transf.keep_atleast(human_datamatrix, minimum=2) # Keep features that appear in at least two samples

# For missing value imputation based on constants relative to the minimum of each feature instead of the full dataset
def fillna_frac_feat_min(df, fraction=0.2):
    """Set NaN to a fraction of the minimum value in each column of the DataFrame."""

    minimum = df.min(axis=0) * fraction
    return df.fillna(minimum)
    
human_datamatrix_I = fillna_frac_feat_min(human_datamatrix, fraction=0.2)
human_datamatrix_N = transf.normalize_PQN(human_datamatrix_I, ref_sample='mean')
human_datamatrix_treated = transf.pareto_scale(transf.glog(human_datamatrix_N, lamb=None))

# Linear interpolation of data
data_treated = pd.read_csv('HD_generated_data_NGP.csv')
data_treated = data_treated.set_index(data_treated.columns[0])
data_treated

with open('HD_data_NGP_lbls.txt') as a:
    lbls = a.read().split('\n')[:-1]

print('\n Data was read and treated')

# Get distribution of intensity values of the dataset
hist = np.histogram(human_datamatrix_treated.values.flatten(), bins=100)
input_realdata_dist = stats.rv_histogram(hist)

print('Data was transformed')

def generator_model(len_input, len_output, n_hidden_nodes, n_labels):
    "Make the generator model of CWGAN-GP."

    data_input = tf.keras.Input(shape=(len_input,), name='data') # Take intensity input
    label_input = tf.keras.Input(shape=(1,), name='label') # Take Label Input

    # Treat label input to concatenate to intensity data after
    label_m = tf.keras.layers.Embedding(n_labels, 30, input_length=1)(label_input)
    label_m = tf.keras.layers.Dense(256, activation='linear', use_bias=True)(label_m)
    #label_m = tf.keras.layers.Reshape((len_input,1,))(label_m)
    label_m2 = tf.keras.layers.Reshape((256,))(label_m)

    joined_data = tf.keras.layers.Concatenate()([data_input, label_m2]) # Concatenate intensity and label data
    # Hidden Dense Layer and Normalization
    joined_data = tf.keras.layers.Dense(n_hidden_nodes, activation=tf.nn.leaky_relu, use_bias=True)(joined_data)
    joined_data = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, use_bias=True)(joined_data)
    joined_data = tf.keras.layers.BatchNormalization()(joined_data)

    # Output - number of features of sample to make
    output = tf.keras.layers.Dense(len_output, activation='linear', use_bias=True)(joined_data)
    
    generator = tf.keras.Model(inputs=[data_input, label_input], outputs=output)
    
    return generator

def critic_model(len_input, n_hidden_nodes, n_labels):
    "Make the critic model of CWGAN-GP."
    
    label_input = tf.keras.Input(shape=(1,)) # Take intensity input
    data_input = tf.keras.Input(shape=(len_input,)) # Take Label Input
    #data_input = tf.keras.layers.Reshape((len_input,1,))(data_input)

    # Treat label input to concatenate to intensity data after
    label_m = tf.keras.layers.Embedding(n_labels, 30, input_length=1)(label_input)
    label_m = tf.keras.layers.Dense(256, activation='linear', use_bias=True)(label_m)
    #label_m = tf.keras.layers.Reshape((len_input,1,))(label_m)
    label_m = tf.keras.layers.Reshape((256,))(label_m)

    joined_data = tf.keras.layers.Concatenate()([data_input, label_m]) # Concatenate intensity and label data
    # Hidden Dense Layer
    joined_data = tf.keras.layers.Dense(n_hidden_nodes, activation=tf.nn.leaky_relu, use_bias=True)(joined_data)
    joined_data = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu, use_bias=True)(joined_data)
    joined_data = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, use_bias=True)(joined_data)
    #joined_data = tf.keras.layers.BatchNormalization()(joined_data)

    # Output Layer - 1 node for critic decision
    output = tf.keras.layers.Dense(1, activation='linear', use_bias=True)(joined_data)
    
    critic = tf.keras.Model(inputs=[data_input, label_input], outputs=output)

    return critic

def generate_predictions(model, num_examples_to_generate, len_input, input_dist, uni_lbls):
    "Generate sample predictions based on a Generator model."
    
    test_input =  tf.constant(input_dist.rvs(size=len_input*num_examples_to_generate), shape=[
        num_examples_to_generate,len_input]) 
    
    if len(uni_lbls) < 3:
        test_labels = tf.constant([1.0]*(num_examples_to_generate//2) + [0.0]*(num_examples_to_generate//2), 
                                  shape=(num_examples_to_generate,1))
    else:
        test_labels = np.array(pd.get_dummies([i for i in range(len(uni_lbls))]*(num_examples_to_generate//len(uni_lbls))))
    predictions = model([test_input, test_labels], training=False) # `training` is set to False.
    return predictions


def training_montage(train_data_o, train_lbls, test_data, test_lbls,
                     epochs, generator, critic, generator_optimizer, critic_optimizer, input_dist,
                    batch_size, grad_pen_weight=10, k_cov_den=50, k_crossLID=15):
    """Train a generator and critic of CWGAN-GP.
    
       Receives training data and respective class labels (train_data_o and train_lbls) and trains a generator and a critic
    model (generator, critic) over a number of epochs (epochs) with a set batch size (batch_size) with the respective 
    optimizers and learning rate (generator_optimizer, critic_optimizer). Gradient Penalty is calculated with grad_pen_weight
    as the weight of the penalty.
       The functions returns at time intervals three graphs to evaluate the progression of the models (Loss plots, coverage,
    density, crossLID and correct first cluster plots and PCA plot with generated and test data). To this end, samples need
    to be generated requiring the distribution to sample the initial input values from (input_dist), and test data and
    respective labels has to be given (test_data and test_lbls). Finally the number of neighbors to consider for 
    coverage/density and crossLID calculation is also needed (k_cov_den, k_crossLID).
    
       train_data_o: Pandas DataFrame with training data;
       train_lbls: List with training data class labels;
       test_data: Pandas DataFrame with test data to evaluate the model;
       test_lbls: List with test data class labels to evaluate the model;
       epochs: Int value with the number of epochs to train the model;
       generator: tensorflow keras.engine.functional.Functional model for the generator;
       critic: tensorflow keras.engine.functional.Functional model for the critic;
       generator_optimizer: tensorflow keras optimizer (with learning rate) for generator;
       critic_optimizer: tensorflow keras optimizer (with learning rate) for critic;
       input_dist: scipy.stats._continuous_distns.rv_histogram object - distribution to sample input values for generator;
       batch_size: int value with size of batch for model training;
       grad_pen_weight: int value (default 10) for penalty weight in gradient penalty calculation;
       k_cov_den: int value (default 50) for number of neighbors to consider for coverage and density calculation in generated
        samples evaluation;
       k_crossLID: int value (default 15) for number of neighbors to consider for crossLID calculation in generated samples
        evaluation.
    """
    
    # Obtaining the train data, randomize its order and divide it be twice the standard deviation of its values
    all_data = train_data_o.iloc[
        np.random.RandomState(seed=145).permutation(len(train_data_o))]/(2*train_data_o.values.std())
    
    # Same treatment for the test data
    test_data = (test_data/(2*test_data.values.std())).values
    training_data = all_data
    train_data = all_data.values
    
    # Change class labels to numerical values while following the randomized ordered of samples
    if len(set(train_lbls)) < 3: # 1 and 0 for when there are only two classes
        train_labels = pd.get_dummies(
            np.array(train_lbls)[np.random.RandomState(seed=145).permutation(len(train_data))]).values[:,0]
        #test_labels = pd.get_dummies(np.array(test_lbls)).values[:,0]
    else: # One hot encoding for when there are more than two classes
        train_labels = pd.get_dummies(
            np.array(train_lbls)[np.random.RandomState(seed=145).permutation(len(train_data))]).values
        #test_labels = pd.get_dummies(np.array(test_lbls)).values
    # Save the order of the labels
    ordered_labels = pd.get_dummies(
            np.array(train_lbls)[np.random.RandomState(seed=145).permutation(len(train_data_o))]).columns

    batch_divisions = int(batch_size / len(set(train_lbls))) # See how many samples of each class will be in each batch
    n_steps = epochs * int(training_data.shape[0] / batch_size) # Number of steps: nº of batches per epoch * nº of epochs
    n_critic = 5

    #update1 = n_steps//200
    update2 = n_steps//20

    i = 0

    for step in range(n_steps):
        
        # Critic Training
        crit_loss_temp = []
        
        # Select real samples for this batch on training and order samples to put samples of the same class together
        real_samp = train_data[i*batch_size: (i+1)*batch_size]
        real_lbls = train_labels[i*batch_size: (i+1)*batch_size]

        real_samples = np.empty(real_samp.shape)
        real_labels = np.empty(real_lbls.shape)
        a = 0
        if len(set(train_lbls)) < 3:
            for l,s in sorted(zip(real_lbls, real_samp), key=lambda pair: pair[0], reverse=True):
                real_samples[a] = s
                real_labels[a] = l
                a = a+1
        else:
            for l,s in sorted(zip(real_lbls, real_samp), key=lambda pair: np.argmax(pair[0]), reverse=False):
                real_samples[a] = s
                real_labels[a] = l
                a = a+1

        for _ in range(n_critic): # For each step, train critic n_critic times
            
            # Generate input for generator
            artificial_samples = tf.constant(input_dist.rvs(size=all_data.shape[1]*batch_size),
                                             shape=[batch_size, all_data.shape[1]])
            artificial_labels = real_labels.copy()

            # Generate artificial samples from the latent vector
            artificial_samples = generator([artificial_samples, artificial_labels], training=True)
            
            with tf.GradientTape() as crit_tape: # See the gradient for the critic

                # Get the logits for the generated samples
                X_artificial = critic([artificial_samples, artificial_labels], training=True)
                # Get the logits for the real samples
                X_true = critic([real_samples, real_labels], training=True)

                # Calculate the critic loss using the generated and real sample results
                c_cost = critic_loss_wgan(X_true, X_artificial)

                # Calculate the gradient penalty
                grad_pen = gradient_penalty_cwgan(batch_size, real_samples, artificial_samples,
                                                  real_labels, artificial_labels, critic)
                # Add the gradient penalty to the original discriminator loss
                crit_loss = c_cost + grad_pen * grad_pen_weight
                #print(crit_loss)
                #crit_loss = c_cost
                
            crit_loss_temp.append(crit_loss)

            # Calculate and apply the gradients obtained from the loss on the trainable variables
            gradients_of_critic = crit_tape.gradient(crit_loss, critic.trainable_variables)
            #print(gradients_of_critic)
            critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

        i = i + 1
        if (step+1) % (n_steps//epochs) == 0:
            i=0

        crit_loss_all.append(np.mean(crit_loss_temp))
        
        # Generator Training
        # Generate inputs for generator, values and labels
        artificial_samples = tf.constant(input_dist.rvs(size=all_data.shape[1]*batch_size), shape=[
                batch_size,all_data.shape[1]])
        
        if len(set(train_lbls)) < 3:
            artificial_labels = tf.constant([1.0]*(batch_size//2) + [0.0]*(batch_size//2), shape=(batch_size,1))
        else:
            artificial_labels = np.array(pd.get_dummies([i for i in range(len(set(train_lbls)))]*batch_divisions))
    
        with tf.GradientTape() as gen_tape: # See the gradient for the generator
            # Generate artificial samples
            artificial_samples = generator([artificial_samples, artificial_labels], training=True)
            
            # Get the critic results for generated samples
            X_artificial = critic([artificial_samples, artificial_labels], training=True)
            # Calculate the generator loss
            gen_loss = generator_loss_wgan(X_artificial)

        # Calculate and apply the gradients obtained from the loss on the trainable variables
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        #print(gradients_of_generator)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        gen_loss_all.append(gen_loss)
            
        if (step + 1) % update2 == 0:

            saved_predictions.append(generate_predictions(generator, 96, all_data.shape[1], 
                                                          input_dist, ordered_labels))
            # See density and coverage and crossLID (divided by 50 to be in the same order as the rest) of latest predictions
            den, cov = gem.evaluation_coverage_density(test_data, saved_predictions[-1], k= k_cov_den, metric='euclidean')
            clid = gem.cross_LID_estimator_byMLE(test_data, saved_predictions[-1], k=k_crossLID, metric='euclidean')/50
            density.append(den)
            coverage.append(cov)
            crossLID.append(clid)

            # Hierarchical clustering of the latest predictions and testing data, 
            # saving the correct 1st cluster fraction results
            dfs_temp = np.concatenate((test_data, saved_predictions[-1].numpy()))
            temp_lbls = ['real']*len(test_data) + ['gen']*len(saved_predictions[-1])
            hca_results = gem.perform_HCA(dfs_temp, temp_lbls, metric='euclidean', method='average')
            corr1stcluster.append(hca_results['correct 1st clustering'])



gen_n = int(gen_nodes_str)

for crit_n_str in test_nodes:

    gen_loss_all = []
    crit_loss_all = []
    saved_predictions = []
    coverage = []
    density = []
    crossLID = []
    corr1stcluster = []

    crit_n = int(crit_n_str)
    generator_optimizer = tf.keras.optimizers.RMSprop(1e-4)
    critic_optimizer = tf.keras.optimizers.RMSprop(1e-4)

    generator = generator_model(data_treated.shape[1], data_treated.shape[1], gen_n, 2)
    critic = critic_model(data_treated.shape[1], crit_n, 2)

    batch_size = 32

    n_epochs = 10

    training_montage(data_treated, lbls, data_treated, lbls, n_epochs, generator,
                        critic, generator_optimizer, critic_optimizer,
                        input_realdata_dist, batch_size, grad_pen_weight=5, k_cov_den=20, k_crossLID=15)
