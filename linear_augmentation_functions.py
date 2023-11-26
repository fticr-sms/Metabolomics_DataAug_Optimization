
# Python File to include functions related to the Linear Augmentation of Data.

# Needed imports

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.stats as stats
import warnings

from collections import Counter


# Linear Augmentation of 'intensity' data

def artificial_sample_linear_generator(sample1, sample2, rnd=None):
    """Takes 2 samples and creates another based on a linear combination of the two samples.
       rnd represents the 'fraction' of sample 1 contribution to the artificial sample. 
    If None, it is randomized between 0 and 1.
       
       Returns pandas Series artificial sample.
    """
    if rnd == None: # If a value between 0 and 1 is not chosen
        rnd = np.random.random(1)[0] # Randomly choosing a value between 0 and 1 to multiply the vector
    elif rnd >= 1 or rnd <= 0:
        raise ValueError('rnd must be a scalar between 0 and 1.')
        
    # Difference between the 2 samples
    vector = sample2 - sample1
    new_sample = (sample1 + rnd * vector)
    
    return new_sample


def artificial_sample_semi_linear_generator(base_sample, sample1, sample2, rnd=None):
    """Takes 3 samples and creates another based on the base sample and the difference between the other two samples.
       rnd represents the 'fraction' of sample 1 contribution to the artificial sample. 
    If None, it is randomized between 0 and 1.
       
       Returns pandas Series artificial sample.
    """
    if rnd == None: # If a value between 0 and 1 is not chosen
        rnd = np.random.random(1)[0] # Randomly choosing a value between 0 and 1 to multiply the vector
    elif rnd >= 1 or rnd <= 0:
        raise ValueError('rnd must be a scalar between 0 and 1.')
        
    # Difference between the 2 samples
    vector = sample2 - sample1
    new_sample = (base_sample + rnd * vector)
    
    return new_sample


# Augmentation of Binary / Feature Occurrence Data

def artificial_binary_sample_generator(sample1, sample2, method='random sampling', rnd=None, binary_rnd_state=None):
    """Takes 2 binary samples and creates another artificial with one of two methods: 'random sampling' and 'fixed sampling'.
       
       In 'random sampling', a threshold ('rnd') is set and a number between 0 and 1 is generated randomly (seeded with
       binary_rnd_state) for each feature in the sample. If the number is below the threshold, binary value is taken from
       sample 1; if not, binary value is taken from sample 2.
       
       In 'fixed sampling', a fraction ('rnd') is set. The features with different values between sample 1 and sample 2 are
       selected. From those, a (1 - 'rnd') fraction of them (rounded) are randomly chosen (seeded with binary_rnd_state) and
       are given the binary value in sample 2. The remaining features are given the binary value in sample 1.
       
       Returns pandas Series binary artificial sample.
    """
    if rnd == None: # If a value between 0 and 1 is not chosen
        rnd = np.random.random(1)[0] # Randomly choosing a value between 0 and 1
    elif rnd >= 1 or rnd <= 0:
        raise ValueError('rnd must be a scalar between 0 and 1.')
    
    rng = np.random.default_rng(binary_rnd_state) # Set a constant rng generator (if chosen)
    new_sample = sample1.copy() # Generate new sample initially equal to sample 1. Thus, only features that will take from
    # sample 2 need to be changed
    
    if method == 'random sampling':
        generated_coefs = rng.random(len(sample1)) # Generating a number for each feature in the sample
        for i in range(len(new_sample)):
            # If number for the feature above the threshold, change the feature with the binary value from sample 2
            if generated_coefs[i] > rnd:
                new_sample.iloc[i] = sample2.iloc[i]
    
    elif method == 'fixed sampling':
        # Set up list with indices of features with different values in sample 1 and sample 2
        differences = []
        for i in range(len(sample1)):
            if sample1.iloc[i] != sample2.iloc[i]:
                differences.append(i)
                
        difs_favour_sample2 = round(len(differences) * (1-rnd)) # Number of features to be given the value of sample 2
        
        # Randomly choose the features to be given the value of sample 2 from the indices in differences
        idx_to_change = rng.choice(differences, size=difs_favour_sample2, replace=False)
        new_sample.iloc[idx_to_change] = sample2.iloc[idx_to_change]
    
    else:
        raise ValueError("method must be either 'random sampling' or 'fixed sampling'. No other methods accepted.")
        
    new_sample.name = None # Reset the name of the series to None
        
    return new_sample

# Functions to induce noise to Feature Occurrence Data, different methods, function imbalanced_noise_inducer recommended

def noise_inducer(binary_data, noise_fraction=0.05, method='random sampling', rnd_state=None):
    """Introduce noise into binary data changing noise_fraction of the columns.
    
       binary_data: pandas Series with binary data.
       noise_fraction: scalar between 0 and 1; approximated fraction of features to be affected by the noise_inducer.
       method: str (default: 'random sampling'); method used to induce noise ('random sampling', 'fixed sampling' or
    'fixed sampling by value').
       rnd_state: scalar (default: None); input a scalar to create a numpy default_rng seed. If none, no seed is given.
       
       returns: pandas Series with noise induced binary data.
    """
    binary_data = binary_data.copy()
    
    if noise_fraction >= 1 or noise_fraction <= 0:
        raise ValueError('noise_fraction must be a scalar between 0 and 1.')
    
    rng = np.random.default_rng(rnd_state) # Set a constant rng generator (if chosen)

    if method == 'random sampling':
        generated_coefs = rng.random(len(binary_data)) # Generating a number for each feature in the sample
        for i in range(len(binary_data)):
            # If number for the feature above the threshold, change the feature with the binary value from sample 2
            if generated_coefs[i] < noise_fraction:
                binary_data.iloc[i] = 1 - binary_data.iloc[i]
                    
    elif method == 'fixed sampling':
        # Randomly choose noise_fraction of the features to be given the value of sample 2 from the indices in differences
        n_feats_change = round(len(binary_data) * noise_fraction) # Nº of features to change
        
        idx_to_change = rng.choice(len(binary_data), size=n_feats_change, replace=False)
        binary_data.iloc[idx_to_change] = 1 - binary_data.iloc[idx_to_change]
        
    elif method == 'fixed sampling by value':
        # Randomly choose noise_fraction of the features with 1 and change to 0 and noise_fraction of the features with 0
        # and change to 1
        ones = binary_data[binary_data == 1]
        zeros = binary_data[binary_data == 0]
        
        n_one_feats_change = round(len(ones) * noise_fraction) # Nº of features with 1 to change
        n_zero_feats_change = round(len(zeros) * noise_fraction) # Nº of features with 0 to change
        
        # Indices names to change with 1 and with 0
        idx_to_change_one = ones.iloc[rng.choice(len(ones), size=n_one_feats_change, replace=False)].index
        idx_to_change_zero = zeros.iloc[rng.choice(len(zeros), size=n_zero_feats_change, replace=False)].index
        
        idx_to_change = list(idx_to_change_one) + list(idx_to_change_zero) # Join list of indices
        # Change 1's to 0's and 0's to 1's in the right indices
        binary_data.loc[idx_to_change] = 1 - binary_data.loc[idx_to_change]
    
    else:
        raise ValueError(
         "method must be either 'random sampling', 'fixed sampling' or 'fixed sampling by value'. No other methods accepted.")
        
    return binary_data


def imbalanced_noise_inducer(binary_data, noise_fraction=0.05, method='random sampling', rnd_state=None,
                            noise_fraction_changer=False, change_factor=2, data=None):
    """Introduce noise into imbalanced binary data changing noise_fraction of the features with 1 values and a similar
    absolute number of features with 0 values.
    
       binary_data: pandas Series with binary data.
       noise_fraction: scalar between 0 and 1; approximated fraction of features to be affected by the noise_inducer.
       method: str (default: 'random sampling'); method used to induce noise ('random sampling' or 'fixed sampling').
       rnd_state: scalar (default: None); input a scalar to create a numpy default_rng seed. If none, no seed is given.
       noise_fraction_changer: bool (default: True); if True, randomly chooses zeros or ones and changes the corresponding
    noise fraction by the change_factor. If '1' or '0', chooses 1 or 0, respectively, to change the noise fraction.
       change_factor: scalar (default: 2); factor by which one of the noise_fractions is multiplied (must be positive).
       data: pandas DataFrame; df with the full original data.
       
       returns: pandas Series with noise induced binary data.
    """
    binary_data = binary_data.copy()
    
    if noise_fraction >= 1 or noise_fraction <= 0:
        raise ValueError('noise_fraction must be a scalar between 0 and 1.')
    elif change_factor < 0:
        raise ValueError('change_factor must be a positive scalar.')
    
    # Setting cc that determines if the noise fraction for 1 or 0 values will be changed
    if noise_fraction_changer:
        if noise_fraction_changer == '1':
            cc = 1
        elif noise_fraction_changer == '0':
            cc = 0
        else:
            cc = np.random.random()
    
    # Obtain the relative probability for a peak to be changed based on the number of times that peak appears in samples if
    # a DataFrame with the original data is passed. Otherwise, equal relative probability between all peaks is given 
    if type(data) != type(None):
        # Normal distribution with center equal to half of the samples of the dataset
        samp_count = data.sum(axis=0)
        mu = int(np.max(samp_count))/2
        std = np.std(samp_count)
        data_pdf = stats.norm(mu,std).pdf(samp_count) # Get a pdf value for each peak
    else:
        data_pdf = stats.uniform(int(np.min(binary_data)), int(np.max(binary_data))+1).pdf(binary_data)
    
    rng = np.random.default_rng(rnd_state) # Set a constant rng generator (if chosen)

    # Randomly choose noise_fraction of the features with 1 and change to 0 and noise_fraction_zero of the features 
    # with 0 and change to 1
    # Separate the data with 1 and 0
    ones = binary_data[binary_data == 1]
    zeros = binary_data[binary_data == 0]

    # In the same way, separate the corresponding probabilities and normalize them (so they add to 1)
    ones_pdf = data_pdf[binary_data == 1]
    zeros_pdf = data_pdf[binary_data == 0]
    ones_pdf = ones_pdf/ones_pdf.sum()
    zeros_pdf = zeros_pdf/zeros_pdf.sum()

    # Change noise_fraction_zero so it takes into account its prevalence in binary_data compared to one values.
    noise_fraction_zero = noise_fraction / ((len(binary_data) - binary_data.sum()) / binary_data.sum())

    if noise_fraction_changer:
        if cc >= 1/2:
            noise_fraction = noise_fraction * change_factor
            if noise_fraction > 1:
                noise_fraction = 1
        else:
            noise_fraction_zero = noise_fraction_zero * change_factor
            if noise_fraction_zero > 1:
                noise_fraction_zero = 1

    if method == 'random sampling':
        generated_coefs = rng.random(len(ones)) # Generating a number for each feature in the sample with value 1
        n_one_feats_change = (generated_coefs < noise_fraction).sum() # Nº of features with 1 to change

        generated_coefs = rng.random(len(zeros)) # Generating a number for each feature in the sample with value 0
        n_zero_feats_change = (generated_coefs < noise_fraction_zero).sum() # Nº of features with 0 to change

    elif method == 'fixed sampling':
        n_one_feats_change = round(len(ones) * noise_fraction) # Nº of features with 1 to change
        n_zero_feats_change = round(len(zeros) * noise_fraction_zero) # Nº of features with 0 to change

    else:
        raise ValueError(
         "method must be either 'random sampling' or 'fixed sampling'. No other methods accepted.")

    # Indices names to change with 1 and with 0
    idx_to_change_one = ones.iloc[rng.choice(len(ones), size=n_one_feats_change, replace=False, p=ones_pdf)].index
    idx_to_change_zero = zeros.iloc[rng.choice(len(zeros), size=n_zero_feats_change, replace=False, p=zeros_pdf)].index

    idx_to_change = list(idx_to_change_one) + list(idx_to_change_zero) # Join list of indices
    # Change 1's to 0's and 0's to 1's in the right indices
    binary_data.loc[idx_to_change] = 1 - binary_data.loc[idx_to_change]
        
    return binary_data


def imbalanced_noise_inducer_middle(binary_data, noise_fraction=0.05, method='random sampling', rnd_state=None,
                            noise_fraction_changer=False, change_factor=2, data=None):
    """Introduce noise into imbalanced binary data changing noise_fraction of the features with 1 values and a similar
    absolute number of features with 0 values.
    
       binary_data: pandas Series with binary data.
       noise_fraction: scalar between 0 and 1; approximated fraction of features to be affected by the noise_inducer.
       method: str (default: 'random sampling'); method used to induce noise ('random sampling' or 'fixed sampling').
       rnd_state: scalar (default: None); input a scalar to create a numpy default_rng seed. If none, no seed is given.
       noise_fraction_changer: bool (default: True); if True, randomly chooses zeros or ones and changes the corresponding
    noise fraction by the change_factor. If '1' or '0', chooses 1 or 0, respectively, to change the noise fraction.
       change_factor: scalar (default: 2); factor by which one of the noise_fractions is multiplied (must be positive).
       data: pandas DataFrame; df with the full original data.
       
       returns: pandas Series with noise induced binary data.
    """
    binary_data = binary_data.copy()
    
    if noise_fraction >= 1 or noise_fraction <= 0:
        raise ValueError('noise_fraction must be a scalar between 0 and 1.')
    elif change_factor < 0:
        raise ValueError('change_factor must be a positive scalar.')
    
    # Setting cc that determines if the noise fraction for 1 or 0 values will be changed
    if noise_fraction_changer:
        if noise_fraction_changer == '1':
            cc = 1
        elif noise_fraction_changer == '0':
            cc = 0
        else:
            cc = np.random.random()
    
    # If a DataFrame with the original data is passed, select only features that appear in 25-75% of samples.
    # Idea is to make features that consistently appear in a very low/high number of samples not affected by noise.
    # If no data is passed, then this step is ignored.
    if type(data) != type(None):
        samp_count = data.sum(axis=0)
        # A method to select from the 25% and 75% percentile of features based on the # of times they appear in samples
        #first_quartile_thresh = samp_count.sort_values().iloc[int(len(data.columns)//4)]
        #third_quartile_thresh = samp_count.sort_values().iloc[int(len(data.columns)//(4/3))]
        
        # Select only features that appear in 25-75% of samples.
        first_quartile_thresh = int(len(data)//4) # 1/4 of all samples
        third_quartile_thresh = int(len(data)//(4/3)) # 3/4 of all samples

        filt_data = samp_count[samp_count.between(first_quartile_thresh, third_quartile_thresh, inclusive='neither')]
    else:
        filt_data = data
    
    rng = np.random.default_rng(rnd_state) # Set a constant rng generator (if chosen)

    # Randomly choose noise_fraction of the features with 1 and change to 0 and noise_fraction_zero of the features 
    # with 0 and change to 1
    # Separate the data with 1 and 0
    ones = binary_data[binary_data == 1]
    zeros = binary_data[binary_data == 0]

    # Select the features with 1 or with 0, respectively, that are in th filt_data created above
    ones_filt_data = ones[np.intersect1d(ones.index, filt_data.index)]
    zeros_filt_data = zeros[np.intersect1d(zeros.index, filt_data.index)]

    # Change noise_fraction_zero so it takes into account its prevalence in binary_data compared to one values.
    noise_fraction_zero = noise_fraction / ((len(zeros_filt_data)) / len(ones_filt_data))

    if noise_fraction_changer:
        if cc >= 1/2:
            noise_fraction = noise_fraction * change_factor
            if noise_fraction > 1:
                noise_fraction = 1
        else:
            noise_fraction_zero = noise_fraction_zero * change_factor
            if noise_fraction_zero > 1:
                noise_fraction_zero = 1

    if method == 'random sampling':
        generated_coefs = rng.random(len(ones)) # Generating a number for each feature in the sample with value 1
        n_one_feats_change = (generated_coefs < noise_fraction).sum() # Nº of features with 1 to change

        generated_coefs = rng.random(len(zeros)) # Generating a number for each feature in the sample with value 0
        n_zero_feats_change = (generated_coefs < noise_fraction_zero).sum() # Nº of features with 0 to change

    elif method == 'fixed sampling':
        n_one_feats_change = round(len(ones) * noise_fraction) # Nº of features with 1 to change
        n_zero_feats_change = round(len(zeros) * noise_fraction_zero) # Nº of features with 0 to change

    else:
        raise ValueError(
         "method must be either 'random sampling' or 'fixed sampling'. No other methods accepted.")

    # Indices names to change with 1 and with 0
    idx_to_change_one = ones.iloc[rng.choice(len(ones), size=n_one_feats_change, replace=False)].index
    idx_to_change_zero = zeros.iloc[rng.choice(len(zeros), size=n_zero_feats_change, replace=False)].index

    idx_to_change = list(idx_to_change_one) + list(idx_to_change_zero) # Join list of indices
    # Change 1's to 0's and 0's to 1's in the right indices
    binary_data.loc[idx_to_change] = 1 - binary_data.loc[idx_to_change]
        
    return binary_data



# Generating dataset with artificial samples (selecting the maximum number of samples wanted and augmentation method)

def artificial_dataset_generator(df, labels=None, max_new_samples_per_label=None, k=None,
                                 binary=False, rnd=None, binary_rnd_state=None, rnd_state=None):
    """Generates a dataset with artificial samples from an original dataframe.
    
       df: pandas DataFrame;
       labels: list; list of labels corresponding to each row in the DataFrame, if None, a '' label is given to every row;
       max_new_samples_per_label: scalar or dict; if None, no maximum of new samples is assigned; if a scalar n, a maximum of n
    new samples is are made for each label; if a dict, each label (key) has a corresponding maximum of new samples (value);
       k: scalar; if not None, it represents the number of nearest neighbors each sample will use to create new samples (if
    below then number of samples in a class). Each samples will use all its nearest neighbors which means that repeating
    pairs might occur (example: sample 'b' is a k-nearest neigbors of sample 'a' and vice-versa; samples will be made for
    both pairs a-b and b-a).
       binary: if False, usual artificial_sample_linear_generator is used to generate new samples; if 'random sampling' or
    'random sampling', artificial_binary_sample_generator is used to generate new samples with the respective method.
       rnd: scalar or list of scalars, scalar to set up consistent artificial samples (meaning changes based on the method used
    to generate the artificial samples); if None, it is random; if a list of n scalars, for each pair of samples, n artificial
    samples are built with rnd equal to each of the scalars in the list.
       binary_rnd_state: scalar; scalar to set up consistent artificial samples (meaning changes based on the method used
    to generate the artificial samples); if None, it is random;
       rnd_state: scalar (default: None); scalar to set up consistent sampling of artificial samples if their number exceeds
    max_new_samples_per_label; if None, it is random;
       
       returns: (pandas DataFrame with artificial samples, corresponding list of sample labels)
    """

    artificial_data = pd.DataFrame(columns=df.columns)
    if labels == None:
        labels = ['' for _ in range(len(df.index))]
        
    if k == None:
        k = df.shape[0]

    # Get metadata from df
    unique_labels = list(set(labels))
    n_unique_labels = len(unique_labels)
    n_all_labels = len(labels)
    sample_dict = {lbl: [] for lbl in unique_labels} 
    for i in range(len(labels)):
        sample_dict[labels[i]].append(df.index[i])

    new_labels = []
    new_labels_count = {lbl: 0 for lbl in unique_labels} 
    for unique_lbl in unique_labels:
        # See how many samples there are in the dataset for each unique_label of the dataset
        label_samples = df.loc[sample_dict[unique_lbl]]
        # if len(samples) = 1 - no pair of 2 samples to make a new one
        # Ensuring all combinations of samples are used to create new samples
        if len(label_samples) > 1 and len(label_samples) <= k:
            for sample1 in range(len(label_samples)):           
                sample2 = len(label_samples) - 1
                while sample1 < sample2:
                    # Creating Artificial Samples
                    if type(rnd) != list:
                        if not binary:
                            artificial_data.loc[
                                    'Arti ' + str(label_samples.index[sample1]) + '-' + str(label_samples.index[sample2])
                                ] = artificial_sample_linear_generator(label_samples.iloc[sample1], label_samples.iloc[sample2], 
                                                                       rnd=rnd)

                        elif binary:
                            artificial_data.loc[
                                    'Arti ' + str(label_samples.index[sample1]) + '-' + str(label_samples.index[sample2])
                                ] = artificial_binary_sample_generator(label_samples.iloc[sample1], label_samples.iloc[sample2],
                                                                   method=binary, rnd=rnd, binary_rnd_state=binary_rnd_state)

                        new_labels.append(unique_lbl)
                        
                    else:
                        for random_number in rnd:
                            if not binary:
                                artificial_data.loc[
                                    'Arti ' + str(label_samples.index[sample1]) + '-' + str(label_samples.index[sample2])
                                    + '_rnd' + str(round(random_number,2))
                                ] = artificial_sample_linear_generator(label_samples.iloc[sample1], label_samples.iloc[sample2], 
                                                                       rnd=random_number)

                            elif binary:
                                artificial_data.loc[
                                    'Arti ' + str(label_samples.index[sample1]) + '-' + str(label_samples.index[sample2])
                                    + '_rnd' + str(round(random_number,2))
                                ] = artificial_binary_sample_generator(label_samples.iloc[sample1], label_samples.iloc[sample2],
                                                                       method=binary, rnd=random_number, 
                                                                       binary_rnd_state=binary_rnd_state)

                            new_labels.append(unique_lbl)
                    
                    sample2 = sample2 - 1
        
        # In case you only want to consider the k-nearest-neighbors for oversampling from each sample
        elif len(label_samples) > 1:
            dist_matrix = dist.squareform(dist.pdist(label_samples, metric='euclidean'))
            k_nearest_neighbors_idxs = np.argsort(dist_matrix, axis=1)[:,1:k+1]
            for sample1 in range(len(label_samples)):    
                for sample2 in k_nearest_neighbors_idxs[sample1]:
                    # Creating Artificial Samples
                    if type(rnd) != list:
                        if not binary:
                            artificial_data.loc[
                                    'Arti ' + str(label_samples.index[sample1]) + '-' + str(label_samples.index[sample2])
                                ] = artificial_sample_linear_generator(label_samples.iloc[sample1], label_samples.iloc[sample2], 
                                                                       rnd=rnd)

                        elif binary:
                            artificial_data.loc[
                                    'Arti ' + str(label_samples.index[sample1]) + '-' + str(label_samples.index[sample2])
                                ] = artificial_binary_sample_generator(label_samples.iloc[sample1], label_samples.iloc[sample2],
                                                                   method=binary, rnd=rnd, binary_rnd_state=binary_rnd_state)

                        new_labels.append(unique_lbl)
                        
                    else:
                        warnings.warn('''When selecting k, all k nearest neighbors of all samples will create a pair. 
                        Thus, sample pairs a-b and b-a are possible and might generate the same sample based on the passed rnd values.''')
                        for random_number in rnd:
                            if not binary:
                                artificial_data.loc[
                                    'Arti ' + str(label_samples.index[sample1]) + '-' + str(label_samples.index[sample2])
                                    + '_rnd' + str(round(random_number,2))
                                ] = artificial_sample_linear_generator(label_samples.iloc[sample1], label_samples.iloc[sample2], 
                                                                       rnd=random_number)

                            elif binary:
                                artificial_data.loc[
                                    'Arti ' + str(label_samples.index[sample1]) + '-' + str(label_samples.index[sample2])
                                    + '_rnd' + str(round(random_number,2))
                                ] = artificial_binary_sample_generator(label_samples.iloc[sample1], label_samples.iloc[sample2],
                                                                       method=binary, rnd=random_number, 
                                                                       binary_rnd_state=binary_rnd_state)

                            new_labels.append(unique_lbl)
                    
        # Number of samples added for each unique label
        if unique_lbl == unique_labels[0]:
            new_labels_count[unique_lbl] = len(new_labels)
        else:
            new_labels_count[unique_lbl] = len(new_labels) - sum(new_labels_count.values())

    if max_new_samples_per_label == None:
        return artificial_data, new_labels
    else:
        
        # Building max_sample_dict dictionary
        try:
            max_new_samples_per_label = int(max_new_samples_per_label)
            max_sample_dict = {lbl: max_new_samples_per_label for lbl in unique_labels}
        except TypeError:
            if isinstance(max_new_samples_per_label, dict):
                max_sample_dict = max_new_samples_per_label
            else: 
                raise TypeError("max_new_samples_per_label must be either a scalar or a dict using the labels given as keys.")
        
        # Randomly sampling rows if row per label exceeds the corresponding value in max_sample_dict
        artificial_data, artificial_labels = random_sample_choice(artificial_data, new_labels, max_sample_dict,
                                                                  unique_labels=unique_labels,
                                                                  labels_count=new_labels_count, in_order=True,
                                                                  rnd_state=rnd_state)
        
    return artificial_data, artificial_labels


def random_sample_choice(data, labels, n_samples_per_label, unique_labels=None, labels_count=None, in_order=True,
                         rnd_state=None, return_idxs=False):
    """Takes a DataFrame and corresponding labels and keeps number of samples per label indicated.
    
       If there are less rows/samples than the number indicated for a label, all rows of that label are kept.
       If there are more, a random selection of the number indicated of the rows of that label are kept.
    
       data: pandas DataFrame;
       labels: list; list of labels corresponding to each row in the DataFrame;
       n_samples_per_label: dict; contains values of number of samples to keep in the data for each label key;
       unique_labels: list; list of unique_labels in the data, if None, it is built based on the labels passed;
       labels_count: dict; contains values of number of samples in the data for each label key, if None, it is built based
    on the labels passed;
       in_order: bool (default: True); if True, the samples of the same labels should be in consecutive rows of the DataFrame;
       rnd_state: scalar (default: None); scalar to set up a numpy default_rng seed; if None, it is random;
       return_idxs: bool (default: False); if True; it also returns the idxs from the original data that were kept in the
    DataFrame. If False, it only returns the DataFrame.
       
       returns: (pandas DataFrame, corresponding list of sample labels)
    """
    
    # Build unique_labels and labels_count if not passed
    if unique_labels == None:
        unique_labels = list(set(labels))
    if labels_count == None:
        labels_count = Counter(labels)

    rng = np.random.default_rng(rnd_state) # Set a constant rng generator (if chosen)

    idxs_to_keep = []
    
    # If the samples of every label are all 'grouped' (consecutive rows)
    if in_order:
        current_sample_idx = 0

        for lbl in unique_labels:
            
            if n_samples_per_label[lbl] == None:
                # Keep all available samples
                previous_sample_idx = current_sample_idx
                current_sample_idx += labels_count[lbl]
                idxs_to_keep.extend(list(range(previous_sample_idx, current_sample_idx)))

            elif labels_count[lbl] <= n_samples_per_label[lbl]:
                # Keep all available samples
                previous_sample_idx = current_sample_idx
                current_sample_idx += labels_count[lbl]
                idxs_to_keep.extend(list(range(previous_sample_idx, current_sample_idx)))

            else:
                # Randomly choose the artificial samples
                idx_to_keep = rng.choice(list(range(labels_count[lbl])), size=n_samples_per_label[lbl], replace=False)
                idxs_to_keep.extend(idx_to_keep + current_sample_idx)
                current_sample_idx += labels_count[lbl]

                #print(data.iloc[previous_sample_idx:current_sample_idx].iloc[idx_to_change])
    
    # If they are not grouped
    else:
        # Build a dictionary with the indices of the rows with each label
        sample_dict = {lbl: [] for lbl in unique_labels} 
        for i in range(len(labels)):
            #sample_dict[labels[i]].append(data.index[i])
            sample_dict[labels[i]].append(i)
        
        for lbl in unique_labels:
     
            if n_samples_per_label[lbl] == None:
                # Keep all available samples
                idxs_to_keep.extend(sample_dict[lbl])
                
            elif labels_count[lbl] <= n_samples_per_label[lbl]:
                # Keep all available samples
                idxs_to_keep.extend(sample_dict[lbl])

            else:
                # Randomly choose the artificial samples
                idx_to_keep = rng.choice(sample_dict[lbl], size=n_samples_per_label[lbl], replace=False)
                idxs_to_keep.extend(idx_to_keep)
                #print(artificial_sample_dict[lbl], idx_to_keep)
    
    if return_idxs:
        return data.iloc[idxs_to_keep], [labels[i] for i in idxs_to_keep], idxs_to_keep
    else:
        return data.iloc[idxs_to_keep], [labels[i] for i in idxs_to_keep]