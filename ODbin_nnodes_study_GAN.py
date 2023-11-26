
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
from GAN_functions import softmax
from GAN_functions import gradient_penalty_cwgan_bin
from GAN_functions import critic_loss_wgan
from GAN_functions import generator_loss_wgan

# Testing different learning rates
# This script is to be executed in a slurm array

# indexing the learning rate to use

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

test_nodes = ('126', '256', '512', '1024')

gen_nodes_str = test_nodes[idx]

print(f"\nINFO: Job with array task id {idx} is using {gen_nodes_str} nodes in generator.")

# Reading Data
base_file = pd.read_csv('ST001618_AN002653_res.txt', sep='\t')
base_file = base_file.set_index('Samples')
df = base_file.iloc[:, 1:].replace({0:np.nan})
samples = list(base_file.iloc[:, 0].values)
# Selecting the samples 
selection = []
for i in samples:
    if i.startswith('Phenotype:Study_Pools'):
        selection.append(False)
    else:
        selection.append(True)
df = df.loc[selection]
df_initial = df.copy()

# Creating the list of 'targets' (labels of samples) of the dataset
labels = []
for i in np.array(samples)[selection]:
    labels.append(i.split(':')[1])

# Represents Binary Simplification pre-treatment
def df_to_bool(df):
    "Transforms data into 'binary' matrices."
    return df.mask(df.notnull(), 1).mask(df.isnull(), 0)

# Performs pre-treatment combinations
def compute_transf(df, norm_ref=None, lamb=None):
    "Computes combinations of pre-treatments and BinSim and returns after treatment datasets in a dict."
    
    data = df.copy()
    
    # Imputation of Missing Values
    imputed = transf.fillna_frac_min_feature(data, fraction=0.2)

    # Normalization
    if norm_ref is not None:
        # Normalization by a reference feature
        norm = transf.normalize_ref_feature(imputed, norm_ref, remove=True)
    else:
        # Normalization by the total sum of intensities
        norm = transf.normalize_sum(imputed)
        # Normalization by PQN
        #norm = transf.normalize_PQN(imputed, ref_sample='mean')
    
    # Pareto Scaling and Generalized Logarithmic Transformation
    P = transf.pareto_scale(imputed)
    NP = transf.pareto_scale(norm)
    NGP = transf.pareto_scale(transf.glog(norm, lamb=lamb))
    GP = transf.pareto_scale(transf.glog(imputed, lamb=lamb))
    
    # Store results
    dataset = {}
    dataset['data'] = df

    dataset['BinSim'] = df_to_bool(data)
    dataset['Ionly'] = imputed
    dataset['P'] = P
    dataset['NP'] = NP
    dataset['GP'] = GP
    dataset['NGP'] = NGP
    
    return dataset

df = df.replace({0:np.nan})
p_df = compute_transf(df, norm_ref=None, lamb=None)

df_storage_train = {}
df_storage_test = {}
lbl_storage_train = {}
lbl_storage_test = {}
real_samples = {}

rng = np.random.default_rng(7519)

# Select the samples which will be in the imbalanced and in the test set
permutations = {}
permutations['User'] = list(rng.permutation(np.where(np.array(labels) == 'User')[0]))
permutations['Non-User'] = list(rng.permutation(np.where(np.array(labels) == 'Non-User')[0]))

for i in range(2):
    train_idxs = {}
    test_idxs = {}
    for cl in permutations.keys():

        train_idxs[cl] = list(np.array(permutations[cl])[i*len(permutations[cl])//2: (i+1)*len(permutations[cl])//2])
        test_idxs[cl] = list(np.array(permutations[cl])[: i*len(permutations[cl])//2]) + list(
            np.array(permutations[cl])[(i+1)*len(permutations[cl])//2:])

    train_idxs = train_idxs['User'] + train_idxs['Non-User']
    test_idxs = test_idxs['User'] + test_idxs['Non-User']

    # Create the imbalanced and test set
    df_storage_train[i+1] = p_df['BinSim'].iloc[train_idxs]
    lbl_storage_train[i+1] = list(np.array(labels)[train_idxs])

    df_storage_test[i+1] = p_df['BinSim'].iloc[test_idxs]
    lbl_storage_test[i+1] = list(np.array(labels)[test_idxs])

    real_samples[i+1] = df_storage_test[i+1].copy()

# Linear interpolation of data
data, lbls = laf.artificial_dataset_generator(df_storage_train[1], labels=lbl_storage_train[1],
                                                max_new_samples_per_label=512, binary='random sampling', 
                                                rnd=list(np.linspace(0.2,0.8,3)), 
                                                binary_rnd_state=1276, rnd_state=514)

print('\n Data was read and BinSim was made')

# Get distribution of intensity values of the dataset
hist = np.histogram(real_samples[1].values.flatten(), bins=100)
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

    # Output - number of features of sample to make with 2 values for each feature - one with a probability of the value being 0
    # and the other with a probaiblity of the value being 1.
    output = tf.keras.layers.Dense(len_output*2, activation='linear', use_bias=True)(joined_data)
    output = softmax(output, len_output) # Gives probability between the values 0 and 1 for each feature
    
    generator = tf.keras.Model(inputs=[data_input, label_input], outputs=output)
    
    return generator

def critic_model(len_input, n_hidden_nodes, n_labels):
    "Make the critic model of CWGAN-GP."
    
    label_input = tf.keras.Input(shape=(1,)) # Take intensity input
    data_input = tf.keras.Input(shape=(len_input,2,)) # Take input that has for each feature 2 values - one with a probability 
    # of the value being 0 and the other with a probaiblity of the value being 1.
    #data_input = tf.keras.layers.Reshape((len_input,1,))(data_input)

    # Treat label input to concatenate to intensity data after
    label_m = tf.keras.layers.Embedding(n_labels, 30, input_length=1)(label_input)
    label_m = tf.keras.layers.Dense(256, activation='linear', use_bias=True)(label_m)
    #label_m = tf.keras.layers.Reshape((len_input,1,))(label_m)
    label_m = tf.keras.layers.Reshape((256,))(label_m)
    
    # Flatten the data
    data_m = tf.keras.layers.Reshape((len_input*2,))(data_input)

    joined_data = tf.keras.layers.Concatenate()([data_m, label_m]) # Concatenate intensity and label data
    # Hidden Dense Layer (Normalization worsened results here)
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
    # `training` is set to False.
    test_input =  tf.constant(input_dist.rvs(size=len_input*num_examples_to_generate), shape=[
        num_examples_to_generate,len_input])
    if len(uni_lbls) < 3:
        test_labels = tf.constant([1.0]*(num_examples_to_generate//2) + [0.0]*(num_examples_to_generate//2), 
                                  shape=(num_examples_to_generate,1))
    else:
        test_labels = []
        for i in range(len(uni_lbls)):
            test_labels.extend([i]*(num_examples_to_generate//len(uni_lbls)))
        test_labels = np.array(pd.get_dummies(test_labels))

    predictions = model([test_input, test_labels], training=False)
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

    update2 = n_steps//20

    i=0

    for step in range(n_steps):
        
        # Critic Training
        crit_loss_temp = []
        
        # Select real samples for this batch on training and order samples to put samples of the same class together
        real_samp = train_data[i*batch_size:(i+1)*batch_size]
        real_lbls = train_labels[i*batch_size:(i+1)*batch_size]

        real_samp_2 = np.empty(real_samp.shape)
        real_labels = np.empty(real_lbls.shape)
        a = 0
        if len(set(train_lbls)) < 3:
            for l,s in sorted(zip(real_lbls, real_samp), key=lambda pair: pair[0], reverse=True):
                real_samp_2[a] = s
                real_labels[a] = l
                a = a+1
        else:
            for l,s in sorted(zip(real_lbls, real_samp), key=lambda pair: np.argmax(pair[0]), reverse=False):
                #print(l, np.argmax(l))
                real_samp_2[a] = s
                real_labels[a] = l
                a = a+1
        
        # Transform real samples to have for each feature two values (probability of the value being 0 or 1) to give as an
        # acceptable input to the critic. In this case the probability of being 0 or 1, since the values are already set are
        # 100% (1) or 0% (0)
        real_samples = tf.one_hot(tf.cast(real_samp_2, 'int32'), 2)
        #ones = np.count_nonzero(real_labels == 1)
        #zeros = np.count_nonzero(real_labels == 0)

        for _ in range(n_critic): # For each step, train critic n_critic times
            
            # Generate input for generator
            artificial_samples = tf.constant(input_dist.rvs(size=all_data.shape[1]*batch_size), shape=[
                batch_size,all_data.shape[1]])
            artificial_labels = real_labels.copy()

            # Generate artificial samples from the latent vector
            artificial_samples = generator([artificial_samples, artificial_labels], training=True)
            #print(real_labels.shape)
            
            with tf.GradientTape() as crit_tape: # See the gradient for the critic

                # Get the logits for the generated samples
                X_artificial = critic([artificial_samples, artificial_labels], training=True)
                # Get the logits for the real samples
                #print(real_samples, real_labels)
                X_true = critic([real_samples, real_labels], training=True)

                # Calculate the critic loss using the generated and real sample results
                c_cost = critic_loss_wgan(X_true, X_artificial)

                # Calculate the gradient penalty
                grad_pen = gradient_penalty_cwgan_bin(batch_size, real_samples, artificial_samples,
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
           
            # 2 steps to generate samples: 1) apply the generator; 2) then choose if the value is 0 or 1 for each feature
            # based on the probabilities generated by the generator
            f_samp = generate_predictions(generator, 96, all_data.shape[1], input_realdata_dist, ordered_labels)
            f_samples = tf.argmax(f_samp, f_samp.get_shape().ndims-1)
            saved_predictions.append(f_samples)

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
    generator_optimizer = tf.keras.optimizers.RMSprop(5e-5)
    critic_optimizer = tf.keras.optimizers.RMSprop(5e-5)

    generator = generator_model(data.shape[1], data.shape[1], gen_n, 2)
    critic = critic_model(data.shape[1], crit_n, 2)

    batch_size = 32

    n_epochs = 500

    training_montage(data, lbls, data, lbls, n_epochs, generator,
                        critic, generator_optimizer, critic_optimizer,
                        input_realdata_dist, batch_size, grad_pen_weight=5, k_cov_den=20, k_crossLID=15)

    # Save the generator models' weights.
    generator.save_weights('gan_models/OD_gen_opt_gen_'+ gen_nodes_str + 'crit_' + crit_n_str)

    results = {'gen_loss': gen_loss_all, 'crit_loss': crit_loss_all, 'saved_pred': saved_predictions,
                'coverage': coverage, 'density': density, 'crossLID': crossLID, 'corr1st_cluster': corr1stcluster}

    # Store data (serialize)
    with open('gan_models/OD_results_opt_gen_'+ gen_nodes_str + 'crit_' + crit_n_str + '.pickle', 'wb') as handle:
        pickle.dump(results, handle)
