from collections import namedtuple

import numpy as np
from numpy.random import default_rng
import pandas as pd
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier
import scipy.stats as stats
from sklearn.decomposition import PCA
import sklearn.cluster as skclust
import sklearn.ensemble as skensemble
import random as rd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import (cohen_kappa_score, mean_squared_error, r2_score,
                            adjusted_rand_score, roc_auc_score, roc_curve, auc)
from sklearn.cross_decomposition import PLSRegression


import metabolinks as mtl
import metabolinks.transformations as trans

# Functions present are for the different kinds of multivariate analysis made in the Jupyter Notebooks.

"""Here are compiled the functions developed for specific applications of multivariate analysis (many use the base function from the
scikit-learn Python package) in metabolomics data analysis workflow. The functions are split in the following sub-sections (designed
for a multivariate analysis method):

- Hierarchical Clustering Analysis: calculating Discrimination Distance (dist_discrim), correct first cluster percentage
(correct_1stcluster_fraction), 'ranks' of the iteration nº two samples were clustered together in linkage matrix (mergerank) and
calculating a correlation (similarity) measure between two dendrograms (Dendrogram_Sim).

- K-means Clustering: perform K-means clustering and store results (Kmeans) support function to select x% of "better" k-means
clustering and calculate Discrimination Distance (global and for each group) and adjusted Rand index (Kmeans_discrim).

- Oversampling functions: simple and incomplete SMOTE (fast_SMOTE) - not in use

- Random Forest: building and evaluating (and storing results) Random Forest models from datasets (simple_RF), other methods in disuse
to make and evaluate Random Forest models from a dataset (RF_M3, RF_M4, overfit_RF) and permutation tests of the significance of
predictive accuracy of Random Forest models (permutation_RF).

- PLS-DA: obtaining X_scores from a PLSRegression (PLSscores_with_labels), obtaining the Y response group memberships matrix
(_generate_y_PLSDA), obtaining PLS scores from models built with 1 to n components (optim_PLS), calculating VIP scores for features to
build PLS-DA models (_calculate_vips), building and evaluating (and storing results) PLS-DA models from datasets (model_PLSDA) and
permutation tests of the significance of predictive accuracy of PLS-DA models (permutation_PLSDA).
"""
# -----------------------------
# Univariate analysis methods
# -----------------------------
def computeFC(data, labels):
    # NOTE: labels must be given explicitly, now

    unique_labels = pd.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError('The number of groups in the data is not two')

    locs0 = [i for i, lbl in enumerate(labels) if lbl == unique_labels[0]]
    locs1 = [i for i, lbl in enumerate(labels) if lbl == unique_labels[1]]

    g0means = data.iloc[locs0, :].mean(axis=0)
    g1means = data.iloc[locs1, :].mean(axis=0)

    FC = g0means / g1means
    FC.name = f'FC ({unique_labels[0]} / {unique_labels[1]})'

    log2FC = np.log2(FC)
    log2FC.name = 'log2FC'

    return pd.concat([FC, log2FC], axis=1)

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing.

       From answer in StOvf 
       https://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python"""

    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def compute_pvalues_2groups(data, labels, equal_var=True, alpha=0.05, useMW=False):
    unique_labels = pd.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError('The number of groups in the data is not two')
    locs0 = [i for i, lbl in enumerate(labels) if lbl == unique_labels[0]]
    locs1 = [i for i, lbl in enumerate(labels) if lbl == unique_labels[1]]
    
    pvalues = []
    for i, col in enumerate(data.columns):
        v0, v1 = data.iloc[locs0, i], data.iloc[locs1, i]
        if not useMW:
            tx = stats.ttest_ind(v0, v1, equal_var=equal_var)
        else:
            tx = stats.mannwhitneyu(v0, v1, alternative='two-sided')
        pvalues.append(tx.pvalue)

    pvalues = pd.Series(pvalues, index=data.columns, name='p-value').sort_values()
    adjusted = p_adjust_bh(pvalues.values) # TODO: optionally use other methods
    adjusted = pd.Series(adjusted, index=pvalues.index, name='FDR adjusted p-value')
    return pd.concat([pvalues, adjusted], axis=1)

def compute_FC_pvalues_2groups(normalized, processed,
                               labels=None,
                               equal_var=True,
                               alpha=0.05, useMW=False):
    FC = computeFC(normalized, labels=labels)
    pvalues = compute_pvalues_2groups(processed,
                                      labels=labels,
                                      equal_var=equal_var,
                                      alpha=alpha, useMW=useMW)
    FC = FC.reindex(pvalues.index)
    return pd.concat([pvalues, FC], axis=1)

# ANOVA
# Hopefully, it is fast enough...

def compute_ANOVA_pvalues(dataset, labels):
    unique_labels = pd.unique(labels)
    if len(unique_labels) < 3:
        raise ValueError('The number of groups in the data is less than 3')

    locs = []
    for lbl in unique_labels:
        llbl = [i for i, albl in enumerate(labels) if albl == lbl]
        locs.append(llbl)

    pvalues = []
    for i, col in enumerate(dataset.columns):
        data = [dataset.iloc[lbllocs, i].values for lbllocs in locs]
        _, p = stats.f_oneway(*data)
        pvalues.append(p)

    pvalues = pd.Series(pvalues, index=dataset.columns, name='p-value').sort_values()
    adjusted = p_adjust_bh(pvalues.values) # TODO: optionally use other methods
    adjusted = pd.Series(adjusted, index=pvalues.index, name='FDR adjusted p-value')
    return pd.concat([pvalues, adjusted], axis=1)

### --------- PCA wrapper functions ---------------------

def compute_df_with_PCs(df, n_components=5, whiten=True, labels=None, return_var_ratios=False):
    pca = PCA(n_components=n_components, svd_solver='full', whiten=whiten)
    pc_coords = pca.fit_transform(df)
    var_explained = pca.explained_variance_ratio_[:pca.n_components_]

    # concat labels to PCA coords (in a DataFrame)
    principaldf = pd.DataFrame(pc_coords, index=df.index, columns=[f'PC {i}' for i in range(1, pca.n_components_+1)])
    if labels is not None:
        labels_col = pd.DataFrame(labels, index=principaldf.index, columns=['Label'])
        principaldf = pd.concat([principaldf, labels_col], axis=1)
    if not return_var_ratios:
        return principaldf
    else:
        return principaldf, var_explained

### --------- Hierarchical Clustering Analysis functions ---------------------
def dist_discrim(Z, sample_labels, method='average'):
    """Give a measure of the normalized distance that a group of samples (same label) is from all other samples in HCA.

        This function calculates the distance from a cluster with all samples with the same label to the closest samples using the HCA
    linkage matrix and the labels (in df) of each sample - Discrimination Distance. For each set of samples with the same label, it
    calculates the difference of distances between where the cluster with all the set of samples was formed and the cluster that joins
    those set of samples with another sample. The normalization of this distance is made by dividing said difference by the max
    distance between any two cluster. If the samples with the same label aren't all in the same cluster before any other sample joins
    them, the distance given to this set of samples is zero. It returns the measure of the average normalized distance as well as a
    dictionary with all the calculated distances for all sets of samples (labels).

        Z: ndarray; hierarchical clustering linkage matrix (from scipy.cluster.hierarchical.linkage).
        method: str; Available methods - "average", "median". This is the method to give the normalized discrimination distance measure
        based on the distances calculated for each set of samples.

        Returns: (global_distance, discrimination_distances)
         global_distance: float or None; normalized discrimination distance measure
         discrimination_distances: dict: dictionary with the discrimination distance for each label.
    """

    # Get metadata
    unique_labels = pd.unique(sample_labels)
    n_unique_labels = len(unique_labels)
    ns = len(sample_labels)

    # Create dictionary with number of samples per label
    sample_number = {label: len([lbl for lbl in sample_labels if lbl == label]) for label in unique_labels}
    min_len = min(sample_number.values())
    max_len = max(sample_number.values())

    # to_tree() returns root ClusterNode and ClusterNode list
    _, cn_list = hier.to_tree(Z, rd=True)

    # dists is a dicionary of cluster_id: distance. Distance is fetched from Z
    dists = {cn.get_id(): cn.dist for cn in cn_list[ns:]}

    # Calculate discriminating distances of a set of samples with the same label
    # store in dictionary `discrims`.
    # `total` accumulates total.
    total = 0
    discrims = {label: 0.0 for label in unique_labels}

    # Z[-1,2] is the maximum distance between any 2 clusters
    max_dist = cn_list[-1].dist

    # For each cluster node
    for cn in cn_list:
        i = cn.get_id()
        n_cn = cn.get_count()
        # skip if cluster too short or too long
        if not (min_len <= n_cn <= max_len):
            continue

        labelset = [sample_labels[loc] for loc in cn.pre_order(lambda x: x.id)]
        # get first element
        label0 = labelset[0]

        # check if cluster length == exactly the number of samples of label of 1st element.
        # If so, all labels must also be equal
        if n_cn != sample_number[label0] or labelset.count(label0) != n_cn:
            continue

        # Compute distances
        # find iteration when cluster i was integrated in a larger one - `itera`
        itera = np.where(Z == i)[0][0]
        dif_dist = Z[itera, 2]

        discrims[label0] = (dif_dist - dists[i]) / max_dist # (dist of `itera` - dist of cn) / max_dist (normalizing)
        total += discrims[label0]
        # print(f'\n-----------\ncluster {i}, label set ----> {labelset}')
        # print('discrims ---->', discrims)
        # print('total so far ---->', total)

    # Method to quantify a measure of a global Discrimination Distance for a linkage matrix.
    if method == 'average':
        separaM = total / n_unique_labels
    elif method == 'median':
        separaM = np.median(list(discrims.values()))
        if separaM == 0:
            separaM = None
    else:
        raise ValueError('Method should be one of: ["average", "median"].')
    # print('return values *********')
    # print(separaM)
    # print(discrims)
    # print('************************')

    return separaM, discrims


def correct_1stcluster_fraction(Z, sample_labels):
    """Calculates the fraction of samples whose first cluster was with a cluster of samples with only the same label.

       df: Pandas DataFrame.
       Z: ndarray; hierarchical clustering linkage matrix (from scipy.cluster.hierarchical.linkage).

       returns: scalar; fraction of samples who initial clustered with a cluster of samples with the same label."""

    # Get metadata
    unique_labels = pd.unique(sample_labels)
    n_unique_labels = len(unique_labels)
    ns = len(sample_labels)

    # to_tree() returns root ClusterNode and ClusterNode list
    _, cn_list = hier.to_tree(Z, rd=True)

    # Create dictionary with number of samples per label
    sample_number = {label: len([lbl for lbl in sample_labels if lbl == label]) for label in unique_labels}
    max_len = max(sample_number.values())

    # To get the number of correct first clusters
    correct_clust_n = 0

    # Iterating through the samples
    for i in range(ns):
        # Get the iteration of HCA where the sample was first clustered with another cluster - `itera`
        itera, _ = np.where(Z[:,:2] == i)
        #print(itera, Z[2][itera, 1-pos])

        # Length of cluster made and see if it is bigger than the label with the most samples (just to speed up calculation)
        len_cluster = Z[itera,3]
        #len_cluster = cn_list[ns + int(itera)].get_count()
        if not (len_cluster <= max_len):
            continue

        # Get labels of the cluster `itera` and see if they are all the same
        labelset = [sample_labels[loc] for loc in cn_list[ns + int(itera)].pre_order(lambda x: x.id)]
        # get first element
        label0 = labelset[0]
        if labelset.count(label0) == len_cluster:
            # If they are, sample i's first cluster was correct
            correct_clust_n = correct_clust_n + 1

    # Return fraction of correct first clusters by dividing by the number of samples
    return correct_clust_n/ns


def mergerank(Z):
    """Creates a 'rank' of the iteration number two samples were linked to the same cluster.

       Function necessary for calculation of Baker's Gamma Correlation Coefficient.

       Z: ndarray; hierarchical clustering linkage matrix (from scipy.cluster.hierarchical.linkage).

       Returns: Matrix/ndarray; Symmetrical Square Matrix (dimensions: len(Z)+1 by len(Z)+1), (i,j) position is the iteration
    number sample i and j were linked to the same cluster (higher rank means the pair took more iterations to be linked together).
    """
    nZ = len(Z)
    # Results array
    kmatrix = np.zeros((nZ + 1, nZ + 1)) # nZ + 1 = number of samples

    # Creating initial cluster matrix
    clust = {}
    for i in range(0, nZ + 1):
        clust[i] = (float(i),)

    # Supplementing cluster dictionary with clusters as they are made in hierarchical clustering and filling matrix with the number of
    # the hierarchical clustering iteration where 2 samples were linked together.
    for r in range(0, nZ):

        # if both clusters joined have only one element
        if Z[r, 0] < nZ + 1 and Z[r, 1] < nZ + 1:
            # Place iteration number at which the samples were clustered in the results array
            kmatrix[int(Z[r, 0]), int(Z[r, 1])] = r + 1
            kmatrix[int(Z[r, 1]), int(Z[r, 0])] = r + 1
            # Add to the cluster Dictionary with the elements in the cluster formed at iteration r. - (nZ + 1 + r): (elements)
            clust[nZ + 1 + r] = (Z[r, 0], Z[r, 1], )

        # if one of the clusters joined has more than one element
        else:
            # Add to the cluster Dictionary with the elements in the cluster formed at iteration r. - (nZ + 1 + r): (elements)
            clust[nZ + 1 + r] = (clust[Z[r, 0]] + clust[Z[r, 1]])
            # Place iteration number at which the samples were clustered in the results array for every pair of samples
            # (one in each of the clusters joined)
            for i in range(0, len(clust[Z[r, 0]])):
                for j in range(0, len(clust[Z[r, 1]])):
                    kmatrix[int(clust[Z[r, 0]][i]), int(clust[Z[r, 1]][j])] = r + 1
                    kmatrix[int(clust[Z[r, 1]][j]), int(clust[Z[r, 0]][i])] = r + 1
    return kmatrix


### --------- K-means Clustering Analysis functions ---------------------
def Kmeans(dfdata, n_labels, iter_num, best_fraction):
    """Performs K-means clustering (scikit learn) n times and returns the best x fraction of them (based on their SSE).

       Auxiliary funtion to Kmeans_discrim.
       SSE - Sum of Squared distances each sample and their closest centroid - Function to be minimized by the algorithm (inertia_ in
    the scikit-learn function).

       dfdata: Pandas DataFrame.
       n_labels: integer; number of different labels in the data (number of clusters)
       iter_num: integer; number of different iterations of k-means clustering to perform.
       best_fraction: scalar; fraction of the best Clusterings (based on their SSE) to return.

       returns: Kmean_store, SSE;
        Kmean_store: list of (int(iter_num*best_fraction)+1) K-means clustering ('best') fits (not ordered) and
        SSE: corresponding list of SSEs (inertia) of each fit."""

    # Store results SSEs and Kmeans
    SSE = []
    Kmean_store = []

    # Number of K-means clustering to fit
    for i in range(iter_num):
        Kmean2 = skclust.KMeans(n_clusters=n_labels)
        Kmean = Kmean2.fit(dfdata)  # Fit K-means clustering

        # List of int(iter_num*best_fraction)+1 elements to store the final list of K-means fit
        if i < (int(iter_num*best_fraction)+1):
            SSE.append(Kmean.inertia_)
            Kmean_store.append(Kmean)
        # Replace the 'worst' K-mean fit in the list everytime a better one appears
        elif Kmean.inertia_ < max(SSE):
            SSE[np.argmax(SSE)] = Kmean.inertia_
            Kmean_store[np.argmax(SSE)] = Kmean

    return Kmean_store, SSE


def Kmeans_discrim(df, sample_labels, method='average', iter_num=1, best_fraction=0.1):
    """Gives measure of the Discrimination Distance of each unique group in the dataset and adjusted Rand Index of K-means clustering.

       This function performs n k-means clustering with the default parameters of sklearn.cluster.KMeans with cluster number (equal to
    the number of unique labels of the dataset). For a chosen x% of the best clusterings (based on their SSE), it checks each of the
    clusters formed to see if only and all the samples of a label/group are present. If not, a distance of zero is given to the set of
    labels with a sample present in the cluster. If yes, the Discrimination Distance is the distance between the centroid of the
    samples cluster and the closest centroid normalized by the maximum distance between any 2 cluster centroids. It then returns the
    mean/median of the Discrimination Distances of all groups, a dictionary with each individual Discrimination Distance, the adjusted
    Rand Index of the clustering and the K-means SSE.

       df: Pandas DataFrame.
       method: str (default: "average"); Available methods - "average", "median". This is the method to give the
    normalized Discrimination Distance measure based on the distances calculated for each set of samples.
       iter_num: integer; number of different iterations of K-means clustering to perform.
       best_fraction: scalar; fraction of the best Clusterings (based on their SSE) to return.

       returns: dictionary; dictionary with each key representing a K-means clustering with 4 results each: Discrimination Distance
    measure, dictionary with the Discrimination Distance for each set of samples, adjusted Rand Index and SSE.
    """
    # Get data parts
    # DataParts = namedtuple('DataParts', 'data_matrix labels names features unique_labels')
    #dfdata = df.cdl.data
    unique_labels = pd.unique(sample_labels)
    all_labels = sample_labels
    n_labels = len(unique_labels)
    #sample_number = {label: len(df.cdl.samples_of(label)) for label in unique_labels}
    sample_number = {label: len([lbl for lbl in sample_labels if lbl == label]) for label in unique_labels}

    # Application of the K-means clustering with n_clusters equal to the number of unique labels.
    # Performing K-means clustering iter-num times and returning the best_fraction of them (int(iter_num*best_fraction)+1)
    Kmean_store, SSE = Kmeans(df, n_labels, iter_num, best_fraction)

    Results_store = {}

    # For each of the 'best' K-means clustering
    for num in range(len(Kmean_store)):
        Kmean = Kmean_store[num]

        # Creating dictionary with number of samples for each group
        Correct_Groupings = {label: 0 for label in unique_labels}
        # Making a matrix with the pairwise distances between any 2 clusters.
        distc = dist.pdist(Kmean.cluster_centers_)
        distma = dist.squareform(distc)
        # maximum distance (to normalize discrimination distancces).
        maxi = max(distc)

        # Check if the two conditions are met (all samples in one cluster and only them)
        # Then calculate Discrimination Distance
        for i in unique_labels:
            if (Kmean.labels_[np.where(np.array(all_labels) == i)] == Kmean.labels_[all_labels.index(i)]).sum() == sample_number[i]:
                if (Kmean.labels_ == Kmean.labels_[all_labels.index(i)]).sum() == sample_number[i]:
                    Correct_Groupings[i] = (
                        min(
                            distma[Kmean.labels_[all_labels.index(i)], :][distma[Kmean.labels_[all_labels.index(i)], :] != 0
                                                                        ]
                        )/ maxi
                    )

        # Method to quantify a measure of a global Discrimination Distance for k-means clustering.
        if method == 'average':
            Correct_Groupings_M = np.array(
                list(Correct_Groupings.values())).mean()
        elif method == 'median':
            Correct_Groupings_M = np.median(list(Correct_Groupings.values()))
            if Correct_Groupings_M == 0:
                Correct_Groupings_M = None
        else:
            raise ValueError(
                'Method not recognized. Available methods: "average", "median".')

        rand_index = adjusted_rand_score(all_labels, Kmean.labels_) # Rand index

        # Store results in the dictionary
        Results_store[num] = Correct_Groupings_M, Correct_Groupings, rand_index, SSE[num]

    return Results_store


### --------- Oversampling functions ---------------------
# In disuse - comments may be outdated
# SMOTE oversampling method - very fast and incomplete method (would not work for all datasets well)
def fast_SMOTE(df, binary=False, max_sample=0):
    """Performs a fast oversampling of a set of spectra (specific, function can't be generalized) based on the simplest SMOTE method.

       New samples are artificially made using the formula: New_Sample = Sample1 + random_value * (Sample2 - Sample1), where the 
    random_value is a randomly generated number between 0 and 1. One new sample is made from any combinations of two different samples
    belonging to the same group/label.

       df: DataFrame.
       binary: bool (default - False); indication if the Spectra has binary data and therefore also ensuring the new samples made are
    also binary or if the Spectra has a "normal" non-binary dataset.
       max_sample: integer (default: 0); number of maximum samples for each label. If < than the label with the most amount of
    samples, this parameter is ignored. Samples chosen to be added to the dataset are randomly selected from all combinations of
    two different samples belonging to the same group/label.

    Returns: DataFrame; Table with extra samples with the name 'Arti(Sample1)-(Sample2)'.
    """
    Newdata = df.copy().cdl.erase_labels()

    # Get metadata from df
    n_unique_labels = df.cdl.label_count
    unique_labels = df.cdl.unique_labels
    all_labels = list(df.cdl.labels)
    n_all_labels = len(all_labels)

    nlabels = []
    nnew = {}
    for i in range(n_unique_labels):
        # See how many samples there are in the dataset for each unique_label of the dataset
        #samples = [df.iloc[:,n] for n, x in enumerate(all_labels) if x == unique_labels[i]]
        label_samples = [df.cdl.subset(label=lbl) for lbl in unique_labels]
        if len(label_samples) > 1:
            # if len(samples) = 1 - no pair of 2 samples to make a new one
            # Ensuring all combinations of samples are used to create new samples
            n = len(label_samples) - 1
            for j in range(len(label_samples)):
                m = 0
                while j < n - m:
                    # Difference between the 2 samples
                    Vector = label_samples[n - m] - label_samples[j]
                    random = np.random.random(1) # Randomly choosing a value between 0 and 1 to multiply the vector
                    if binary:
                        # Round values to 0 or 1 so the data stays binary while still creating "relevant" "new" data
                        Newdata[
                            'Arti' + unique_labels[j] + '-' + unique_labels[n - m]
                        ] = round(label_samples[j] + random[0] * Vector)
                    else:
                        Newdata[
                            'Arti' + unique_labels[j] + '-' + unique_labels[n - m]
                        ] = (label_samples[j] + random[0] * Vector)
                    m = m + 1
                    # Giving the correct label to each new sample
                    nlabels.append(unique_labels[i])

        # Number of samples added for each unique label
        if i == 0:
            nnew[unique_labels[i]] = len(nlabels)
        else:
            nnew[unique_labels[i]] = len(nlabels) - sum(nnew.values())

    # Creating dictionary with number of samples for each group
    sample_number = {label: len(df.cdl.samples_of(label)) for label in unique_labels}

    # Choosing samples for each group/labels to try and get max_samples in total of that label.
    if max_sample >= max(sample_number.values()):
        # List with names of the samples chosen for the final dataset
        chosen_samples = list(df.samples)
        nlabels = []
        loca = 0
        for i in unique_labels:
            # Number of samples to add
            n_choose = max_sample - sample_number[i]
            # If there aren't enough new samples to get the total max_samples, choose all of them.
            if n_choose > nnew[i]:
                n_choose = nnew[i]
            # Random choice of the samples for each label that will be added to the original dataset
            chosen_samples.extend(
                rd.sample(
                    list(
                        Newdata.columns[
                            n_all_labels + loca: n_all_labels + loca + nnew[i]
                        ]
                    ),
                    n_choose,
                )
            )
            loca = loca + nnew[i]
            nlabels.extend([i] * n_choose)

        # Creating the dataframe with the chosen samples
        Newdata = Newdata[chosen_samples]

    # Creating the label list for the Pandas DataFrame
    Newlabels = all_labels + nlabels
    Newdata = mtl.add_labels(Newdata, labels=Newlabels)
    return Newdata


### --------- Random Forest functions ---------------------

def RF_model(df, y, return_cv=True, iter_num=1, n_trees=200, cv=None, n_fold=5, **kwargs):
    fitted_model = skensemble.RandomForestClassifier(n_estimators=n_trees)
    fitted_model = fitted_model.fit(df, y)
    if not return_cv:
        return(fitted_model)
    if cv is None:
        cv = StratifiedKFold(n_fold, shuffle=True)
    if iter_num == 1:
        rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
        scores = cross_validate(rf, df, y, cv=cv, **kwargs)
    else:
        scores=[]
        for _ in range(iter_num):
            rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
            cv_res = cross_validate(rf, df, y, cv=cv, **kwargs)
            scores.append(cv_res)
    return {'model': fitted_model, 'cv_scores': scores}


# RF_model_CV - RF application and result extraction.
def RF_model_CV(df, y, iter_num=1, n_fold=5, n_trees=200):
    """Performs stratified k-fold cross validation on Random Forest classifier of a dataset n times giving its accuracy and ordered
    most important features.

       Parameters are estimated by stratified k-fold cross-validation. Iteration changes the random sampling of the k-folds for
    cross-validation.
       Feature importance in the Random Forest models is calculated by the Gini Importance (feature_importances_) of scikit-learn.

       df: Pandas DataFrame.
       y: the target array (following scikit learn conventions)
       iter_num: int (default - 1); number of iterations for CV.
       n_fold: int (default - 5); number of groups to divide dataset in for stratified k-fold cross-validation
            (max n_fold = minimum number of samples belonging to one group).
       n_trees: int (default - 200); number of trees in each Random Forest.

       Returns: (scores, import_features); 
            scores: list of the scores/accuracies of k-fold cross-validation of the random forests
                (one score for each iteration and each group)
            import_features: list of tuples (index number of feature, feature importance, feature name)
                ordered by decreasing feature importance.
    """

    nfeats = df.shape[1]

    # Setting up variables for result storing
    imp_feat = np.zeros((iter_num * n_fold, nfeats))
    accuracy_scores = []
    f = 0

    # Number of times Random Forest cross-validation is made
    # with `n_fold` randomly generated folds.
    for _ in range(iter_num):
        # Use stratified n_fold cross validation
        kf = StratifiedKFold(n_fold, shuffle=True)
        CV_accuracy_scores = []
        # Fit and evaluate a Random Forest model for each fold
        for train_index, test_index in kf.split(df, y):
            # Random Forest setup and fit
            rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
            X_train, X_test = df.iloc[train_index, :], df.iloc[test_index, :]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            rf.fit(X_train, y_train)

            # Compute performance and important features
            CV_accuracy_scores.append(rf.score(X_test, y_test)) # Predictive Accuracy
            imp_feat[f, :] = rf.feature_importances_ # Importance of each feature
            f = f + 1

        # Average Predictive Accuracy in this iteration
        accuracy_scores.append(np.mean(CV_accuracy_scores))

    # Collect and order all important features values from each Random Forest
    imp_feat_sum = imp_feat.sum(axis=0) / (iter_num * n_fold)
    sorted_imp_feat = sorted(enumerate(imp_feat_sum), key=lambda x: x[1], reverse=True)

    # locs are sufficient as a reference to features
    #imp_feat_tuples = [(loc, importance) for loc, importance in sorted_imp_feat]
    
    if iter_num == 1:
        return {'accuracy': accuracy_scores[0], 'important_features': sorted_imp_feat}
    else:
        return {'accuracy': accuracy_scores, 'important_features': sorted_imp_feat}


def RF_ROC_cv(df, y, pos_label, n_fold=5, n_trees=200, n_iter=1):

    # transform y to an array
    target = [lbl==pos_label for lbl in y]
    y = np.array(target, dtype=int)
    X = df.values

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=n_fold, shuffle=True)
    classifier = skensemble.RandomForestClassifier(n_estimators=n_trees)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for _ in range(n_iter):
        for i, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X[train], y[train])
            scores = classifier.predict_proba(X[test])[:, 1]

            fpr, tpr, _ = roc_curve(y[test], scores)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(y[test], scores))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    return {'average fpr': mean_fpr, 'average tpr': mean_tpr, 
            'upper tpr': tprs_upper, 'lower trp': tprs_lower,
            'mean AUC': mean_auc, 'std AUC': std_auc}




# In disuse and outdated - comments also outdated
# Function for method 3 - SMOTE on the training set
def RF_M3(df, y, iter_num=20, binary=False, test_size=0.1, n_trees=200):
    """Builds Random Forest classifiers of a dataset (oversampling the training set) n times giving its predictive accuracy,
    Kappa Cohen score and most important features all estimated by stratified k-fold cross-validation.

       df: DataFrame.
       iter_num: int (default - 20); number of iterations that Random Forests are repeated.
       binary: bool (default - False); indication if the Spectra has binary data and therefore also ensuring the new samples made are
    also binary or if the Spectra has a "normal" non-binary dataset.
       test_size: scalar (default - 0.1); number between 0 and 1 equivalent to the fraction of the samples for the test group.
       n_trees: int (default - 200); number of trees in each Random Forest.

       Returns: (scores, cohen_scores, import_features);
            scores: scalar; mean of the scores of the Random Forests
            cohen_scores: scalar; mean of the Cohen's Kappa score of the Random Forests
            import_features: list of tuples (index number of feature, feature importance, feature name)
                ordered by decreasing feature importance.
    """
    # Get data parts
    # DataParts = namedtuple('DataParts', 'data_matrix labels names features unique_labels')
    all_labels = y
    # n_labels = len(unique_labels)
    # sample_number = {label: len(df.cdl.samples_of(label)) for label in unique_labels}

    imp_feat = np.zeros((iter_num, df.shape[1]))
    cks = []
    scores = []

    for i in range(iter_num):
        # Splitting data and performing SMOTE on the training set.
        X_train, X_test, y_train, y_test = train_test_split(
            df, all_labels, test_size=test_size)

        X_Aligned = X_train
        X_Aligned.cdl.labels = y_train
        Spectra_S = fast_SMOTE(X_Aligned, binary=binary)
        # Random Forest setup and fit.
        rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
        rf.fit(Spectra_S, Spectra_S.cdl.labels)

        # Extracting the results of the random forest model built
        y_pred = rf.predict(X_test)
        imp_feat[i, :] = rf.feature_importances_
        cks.append(cohen_kappa_score(y_test, y_pred))
        scores.append(rf.score(X_test, y_test))

    # Joining and ordering all important features values from each random forest
    imp_feat_sum = imp_feat.sum(axis=0) / iter_num
    imp_feat_sum = sorted(enumerate(imp_feat_sum), key=lambda x: x[1], reverse=True)
    imp_feat_ord = []
    for i, j in imp_feat_sum:
        imp_feat_ord.append((i, j, df.index[i]))

    return np.mean(scores), np.mean(cks), imp_feat_ord


# In disuse and outdated - comments also outdated
# Function for method 4 - SMOTE on the training set and NGP processing of training and test data together.
def RF_M4(df, reffeat, iter_num=20, test_size=0.1, n_trees=200):
    """Builds Random Forest classifiers of a dataset (after oversampling the training sets and data processing both sets) n times
    giving its predictive accuracy, Kappa Cohen score and most important features all estimated by stratified k-fold cross-validation.

       df: DataFrame.
       reffeat: scalar; m/z of the reference feature to normalize the samples.
       iter_num: int (default - 20); number of iterations that Random Forests are repeated.
       binary: bool (default - False); indication if the Spectra has binary data and therefore also ensuring the new samples made are
    also binary or if the Spectra has a "normal" non-binary dataset.
       test_size: scalar (default - 0.1); number between 0 and 1 equivalent to the fraction of the samples for the test group.
       n_trees: int (default - 200); number of trees in each Random Forest.

       Returns: (scores, cohen_scores, import_features);
            scores: scalar; mean of the scores of the Random Forests
            cohen_scores: scalar; mean of the Cohen's Kappa score of the Random Forests
            import_features: list of tuples (index number of feature, feature importance, feature name)
                ordered by decreasing feature importance.
    """
    imp_feat = np.zeros((iter_num, len(df) - 1))
    accuracy = []
    scores = []
    for i in range(iter_num):
        # Splitting data and performing SMOTE on the training set.
        X_train, X_test, _, y_test = train_test_split(
            df.T, df.cdl.labels, test_size=test_size
        )
        X_Aligned = X_train.T
        Spectra_S = fast_SMOTE(X_Aligned, binary=False)

        # NGP processing of the data
        Spectra_S_J = Spectra_S.join(X_test.T)
        Spectra_S_J.labels = Spectra_S.labels + y_test
        Norm_S = trans.normalize_ref_feature(Spectra_S_J, reffeat)
        glog_S = trans.glog(Norm_S)
        Euc_glog_S = trans.pareto_scale(glog_S)
        X_train = Euc_glog_S.data.iloc[:, : -len(y_test)]
        X_test = Euc_glog_S.data.iloc[:, -len(y_test):]

        # Random Forest setup and fit
        rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
        rf.fit(X_train.T, Euc_glog_S.labels[: -len(y_test)])

        # Extracting the results of the Random Forest model built
        y_pred = rf.predict(X_test.T)
        imp_feat[i, :] = rf.feature_importances_
        accuracy.append(cohen_kappa_score(y_test, y_pred))
        scores.append(rf.score(X_test.T, y_test))

    # Joining and ordering all important features values from each Random Forest
    imp_feat_sum = imp_feat.sum(axis=0) / iter_num
    imp_feat_sum = sorted(enumerate(imp_feat_sum), key=lambda x: x[1], reverse=True)
    imp_feat_ord = []
    for i, j in imp_feat_sum:
        imp_feat_ord.append((i, j, df.index[i]))

    return np.mean(scores), np.mean(accuracy), imp_feat_ord


# Test the data with the training data, then check the difference with simple_RF. If this one is much higher, there is clear overfitting
def overfit_RF(Spectra, iter_num=20, test_size=0.1, n_trees=200):
    """Builds Random Forest classifiers of a dataset n times giving its predictive accuracy, Kappa Cohen score and most important
    features all estimated by stratified k-fold cross-validation.

       Spectra: Pandas DataFrame.
       iter_num: int (default - 20); number of iterations that Random Forest are repeated.
       test_size: scalar (default - 0.1); number between 0 and 1 equivalent to the fraction of the samples for the test group.
       n_trees: int (default - 200); number of trees in each Random Forest.

       Returns: (scores, cohen_scores, import_features);
            scores: scalar; mean of the scores of the Random Forests
            cohen_scores: scalar; mean of the Cohen's Kappa score of the Random Forests
            import_features: list of tuples (index number of feature, feature importance, feature name)
                ordered by decreasing feature importance.
    """
    imp_feat = np.zeros((iter_num, len(Spectra)))
    cks = []
    scores = []
    CV = []

    for i in range(iter_num):  # number of times Random Forests are made
        # Random Forest setup and fit
        rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
        # X_train, X_test, y_train, y_test = train_test_split(Spectra.T,
        # Spectra.cdl.labels, test_size = test_size)
        rf.fit(Spectra.T, Spectra.cdl.labels)

        # Extracting the results of the Random Forest model built
        y_pred = rf.predict(Spectra.T)
        imp_feat[i, :] = rf.feature_importances_
        cks.append(cohen_kappa_score(Spectra.cdl.labels, y_pred))
        scores.append(rf.score(Spectra.T, Spectra.cdl.labels))
        CV.append(np.mean(cross_val_score(rf, Spectra.T, Spectra.cdl.labels, cv=3)))

    # Joining and ordering all important features values from each Random Forest
    imp_feat_sum = imp_feat.sum(axis=0) / iter_num
    imp_feat_sum = sorted(enumerate(imp_feat_sum), key=lambda x: x[1], reverse=True)
    imp_feat_ord = []
    for i, j in imp_feat_sum:
        imp_feat_ord.append((i, j, Spectra.index[i]))

    return np.mean(scores), np.mean(cks), imp_feat_ord, np.mean(CV)


def permutation_RF(df, labels, iter_num=100, n_fold=3, n_trees=200):
    """Performs permutation test n times of a dataset for Random Forest classifiers giving its predictive accuracy (estimated by
    stratified 3-fold cross-validation) for the original and all permutations made and respective p-value.

       df: Pandas DataFrame.
       iter_num: int (default - 100); number of permutations made.
       n_fold: int (default - 3); number of groups to divide dataset in for k-fold cross-validation (max n_fold = minimum number of
    samples belonging to one group).
       n_trees: int (default - 200); number of trees in each Random Forest.

       Returns: (scalar, list of scalars, scalar);
        estimated predictive accuracy of the non-permuted Random Forest model
        estimated predictive accuracy of all permuted Random Forest models
        p-value ((number of permutations with accuracy > original accuracy) + 1)/(number of permutations + 1).
    """
    
    # get a bit generator
    rng = default_rng()
    
    # Setting up variables for result storing
    Perm = []
    # List of columns to shuffle and dataframe of the data to put columns in NewC shuffled order
    NewC = np.arange(df.shape[0])
    df = df.copy()

    # For dividing the dataset in balanced n_fold groups with a random random state maintained in all permutations (identical splits)
    kf = StratifiedKFold(
        n_fold, shuffle=True, random_state=np.random.randint(1000000000)
    )


    for _ in range(iter_num + 1):
        # Number of different permutations + original dataset where Random Forest cross-validation will be made
        # Temporary dataframe with columns in order of the NewC
        temp = df.iloc[NewC, :]
        perm = []

        # Repeat for each of the k groups the random forest model fit and classification
        for train_index, test_index in kf.split(df, labels):
            # Random Forest setup and fitting
            rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
            # X_train, X_test = temp[temp.columns[train_index]].T, temp[temp.columns[test_index]].T
            X_train, X_test = temp.iloc[train_index, :], temp.iloc[test_index, :]
            y_train, y_test = (
                [labels[pos] for pos in train_index],
                [labels[pos] for pos in test_index],
            )

            rf.fit(X_train, y_train)
            # Obtaining results with the test group
            perm.append(rf.score(X_test, y_test))

        # Shuffle dataset columns - 1 permutation of the columns (leads to permutation of labels)
        rng.shuffle(NewC)

        # Appending K-fold cross-validation predictive accuracy
        Perm.append(np.mean(perm))

    # Taking out K-fold cross-validation accuracy for the non-shuffled (labels) dataset and p-value calculation
    CV = Perm[0] # Non-permuted dataset results - Perm [0]
    pvalue = (sum(Perm[1:] >= Perm[0]) + 1) / (iter_num + 1)

    return CV, Perm[1:], pvalue


### --------- PLS-DA functions ---------------------
def fit_PLSDA_model(data, labels, n_comp=10, return_scores=True,
                    scale=False, encode2as1vector=True,
                    lv_prefix='LV ', label_name='Label'):
    "Obtain X-scores of a PLSRegression model built from a labelled dataset."
    # create label lists

    unique_labels = list(pd.unique(labels))
    is1vector = (len(unique_labels) == 2) and encode2as1vector

    # Generate the response variable Y for PLSRegression
    target = _generate_y_PLSDA(labels, unique_labels, is1vector)

    plsda = PLSRegression(n_components=n_comp, scale=scale)

    # Fitting the model and getting the X_scores
    plsda.fit(X=data,Y=target)

    if return_scores:
        LV_score = pd.DataFrame(plsda.x_scores_, columns=[f'{lv_prefix}{i}' for i in range(1, n_comp+1)])
        labels_col = pd.DataFrame(labels, columns=[label_name])
        LV_score = pd.concat([LV_score, labels_col], axis=1)
    
    if return_scores:
        return plsda, LV_score
    else:
        return plsda


def _generate_y_PLSDA(all_labels, unique_labels, is1vector):
    "Returns Y response variable for PLS-DA models."
    if not is1vector:
        # Setting up the y matrix for when there are more than 2 classes (multi-class) with one-hot encoding
        matrix = pd.get_dummies(all_labels)
        matrix = matrix[unique_labels]
    else:
        # Create two binary vectors
        matrix = pd.get_dummies(all_labels)
        matrix = matrix[unique_labels]
        # use first column to encode
        matrix = matrix.iloc[:, 0].values  # a numpy array
    return matrix


def optim_PLSDA_n_components(df, labels, encode2as1vector=True, max_comp=50, n_fold=5):
    """Searches for an optimum number of components to use in PLS-DA by accuracy (3-fold cross validation) and mean-squared errors.

    df: DataFrame; X equivalent in PLS-DA (training vectors).
    labels: labels to target
    max_comp: integer; upper limit for the number of components used.
    n_fold: int (default - 3); number of groups to divide dataset in
    for k-fold cross-validation (max n_fold = minimum number of samples
    belonging to one group).

    Returns: (list, list, list), n-fold cross-validation (q2) score and r2 score and mean squared errors for all components searched.
    """
    # Preparating lists to store results
    CVs = []
    CVr2s = []
    MSEs = []
    Accuracy = []
    nright = 0
   
    unique_labels = list(pd.unique(labels))

    is1vector = len(unique_labels) == 2 and encode2as1vector

    matrix = _generate_y_PLSDA(labels, unique_labels, is1vector)

    if is1vector:
        # keep a copy to use later
        target1D = matrix.copy()

    # Repeating for each component from 1 to max_comp
    for i in range(1, max_comp + 1):
        cv = []
        cvr2 = []
        mse = []
        accuracy = []

        # Splitting data into groups for n-fold cross-validation
        kf = StratifiedKFold(n_fold, shuffle=True)

        for train_index, test_index in kf.split(df, labels):
            # NOTE: scaling should have been done, at this point
            plsda = PLSRegression(n_components=i, scale=False)
            X_train, X_test = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()
            if not is1vector:
                y_train = matrix.iloc[train_index, :].copy()
                y_test = matrix.iloc[test_index, :].copy()
            else:
                y_train, y_test = target1D[train_index], target1D[test_index]
                correct = target1D[test_index]

            # Fitting the model
            plsda.fit(X=X_train, Y=y_train)

            # Obtain results with the test group
            y_pred = plsda.predict(X_test)
            cv.append(plsda.score(X_test, y_test))
            cvr2.append(r2_score(plsda.predict(X_train), y_train))
            mse.append(mean_squared_error(y_test, y_pred))

            # Decision rule for classification
            # Decision rule chosen: sample belongs to group where it has max y_pred (closer to 1)
            # In case of 1,0 encoding for two groups, round to nearest integer to compare

            if not is1vector:
                for i in range(len(y_pred)):
                    if list(y_test.iloc[i, :]).index(max(y_test.iloc[i, :])) == np.argmax(
                        y_pred[i]
                    ):
                        nright += 1  # Correct prediction
            else:
                rounded = np.round(y_pred)
                for i in range(len(y_pred)):
                    if rounded[i] == correct[i]:
                        nright += 1  # Correct prediction


            # Calculate accuracy for this iteration
            accuracy.append(nright / len(labels))

        # Storing results for each number of components
        CVs.append(np.mean(cv))
        CVr2s.append(np.mean(cvr2))
        MSEs.append(np.mean(mse))
        Accuracy.append(np.mean(accuracy)) # not used yet...

    return PLSDA_optim_results(CVscores=CVs, CVR2=CVr2s, MSE=MSEs)

PLSDA_optim_results = namedtuple('PLSDA_optim_results', 'CVscores CVR2 MSE')


def _calculate_vips(model):
    """ VIP (Variable Importance in Projection) of the PLSDA model for each variable in the model.

        model: PLS Regression model fitted to a dataset from scikit-learn.

        returns: list; VIP score for each variable from the dataset.
    """
    # Set up the variables
    t = model.x_scores_
    W = model.x_weights_
    q = model.y_loadings_
    n_rows_weights, n_cols_weights = W.shape
    vips = np.empty((n_rows_weights,))

    # Calculate VIPs
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T, t), q.T), q)).reshape(n_cols_weights, -1)
    total_s = np.sum(s)
    for i in range(n_rows_weights):
        weight = np.array([(W[i, j] / np.linalg.norm(W[:, j])) ** 2 for j in range(n_cols_weights)])
        vips[i] = np.sqrt(n_rows_weights * (np.matmul(s.T, weight)) / total_s)

    return vips


def PLSDA_model_CV(df, labels, n_comp=10,
                   n_fold=5,
                   iter_num=1,
                   encode2as1vector=True,
                   feat_type='Coef'):
    
    """Perform PLS-DA with n-fold cross-validation.

    df: pandas DataFrame; includes X equivalent in PLS-DA (training vectors).
    labels: target labels.
    n_comp: integer; number of components to use in PLS-DA.
    n_fold: int (default: 5); number of groups to divide dataset in for k-fold cross-validation
        (NOTE: max n_fold can not exceed minimum number of samples per class).
    iter_num: int (default: 1); number of iterations that cross validation is repeated.
    feat_type: string (default: 'Coef'); types of feature importance metrics to use; accepted: {'VIP', 'Coef', 'Weights'}.

    Returns: (accuracy, n-fold score, r2score, import_features);
        accuracy: list of accuracy values in group selection
        n-fold score : n-fold cross-validation score
        r2score: r2 score of the model
        import_features: list of tuples (index number of feature, feature importance, feature name)
            ordered by decreasing feature importance.
    """
    # Setting up lists and matrices to store results
    CVR2 = []
    accuracies = []
    Imp_Feat = np.zeros((iter_num * n_fold, df.shape[1]))
    f = 0

    unique_labels = list(pd.unique(labels))

    is1vector = len(unique_labels) == 2 and encode2as1vector

    matrix = _generate_y_PLSDA(labels, unique_labels, is1vector)

    if is1vector:
        # keep a copy to use later
        target1D = matrix.copy()

    # Number of iterations equal to iter_num
    for i in range(iter_num):
        # use startified n-fold cross-validation with shuffling
        kf = StratifiedKFold(n_fold, shuffle=True)
        
        # Setting up storing variables for n-fold cross-validation
        nright = 0
        cvr2 = []

        # Iterate through cross-validation procedure
        for train_index, test_index in kf.split(df, labels):
            plsda = PLSRegression(n_components=n_comp, scale=False)
            X_train, X_test = df.iloc[train_index, :], df.iloc[test_index, :]
            if not is1vector:
                y_train = matrix.iloc[train_index, :].copy()
                y_test = matrix.iloc[test_index, :].copy()

            else:
                y_train, y_test = target1D[train_index], target1D[test_index]
                correct = target1D[test_index]

            # Fit PLS model
            plsda.fit(X=X_train, Y=y_train)

            # Obtain results with the test group
            y_pred = plsda.predict(X_test)
            cvr2.append(r2_score(y_test, y_pred))

            # Decision rule for classification
            # Decision rule chosen: sample belongs to group where it has max y_pred (closer to 1)
            # In case of 1,0 encoding for two groups, round to nearest integer to compare
            # if not is1vector:
            #     for i in range(len(y_pred)):
            #         where_max = np.argmax(y_pred[i])

            if not is1vector:
                for i in range(len(y_pred)):
                    if list(y_test.iloc[i, :]).index(max(y_test.iloc[i, :])) == np.argmax(
                        y_pred[i]
                    ):
                        nright += 1  # Correct prediction
            else:
                rounded = np.round(y_pred)
                for i in range(len(y_pred)):
                    if rounded[i] == correct[i]:
                        nright += 1  # Correct prediction

            # Calculate important features (3 different methods to choose from)
            if feat_type == 'VIP':
                Imp_Feat[f, :] = _calculate_vips(plsda)
            elif feat_type == 'Coef':
                Imp_Feat[f, :] = abs(plsda.coef_).sum(axis=1)
            elif feat_type == 'Weights':
                Imp_Feat[f, :] = abs(plsda.x_weights_).sum(axis=1)
            else:
                raise ValueError(
                    'Type not Recognized. Types accepted: "VIP", "Coef", "Weights"'
                )

            f += 1

        # Calculate the accuracy of the group predicted and storing score results
        accuracies.append(nright / len(labels))
        CVR2.append(np.mean(cvr2))

    # Join and sort all important features values from each cross validation group and iteration.
    Imp_sum = Imp_Feat.sum(axis=0) / (iter_num * n_fold)
    imp_features = sorted(enumerate(Imp_sum), key=lambda x: x[1], reverse=True)
    if iter_num == 1:
        return {'accuracy': accuracies[0], 'Q2': CVR2[0], 'important_features': imp_features}
    else:
        return {'accuracy': accuracies, 'Q2': CVR2, 'important_features': imp_features}


def permutation_PLSDA(df, labels, n_comp=10, n_fold=5, iter_num=100, encode2as1vector=True):
    """Performs permutation test n times of a dataset for PLS-DA classifiers giving its predictive accuracy (estimated by
    stratified 3-fold cross-validation) for the original and all permutations made and respective p-value.

       df: DataFrame. Includes X and Y equivalent in PLS-DA (training vectors and groups).
       n_comp: integer; number of components to use in PLS-DA.
       n_fold: int (default - 3); number of groups to divide dataset in
    for k-fold cross-validation (max n_fold = minimum number of samples belonging to one group).
       iter_num: int (default - 100); number of permutations made (times labels are shuffled).

       Returns: (scalar, list of scalars, scalar);
        estimated predictive accuracy of the non-permuted PLS-DA model
        estimated predictive accuracy of all permuted PLS-DA models
        p-value ((number of permutations with accuracy > original accuracy) + 1)/(number of permutations + 1).
    """
    
    # get a bit generator
    rng = default_rng()
    
    # list to store results
    Accuracy = []

    # list of rows to shuffle and dataframe of the data to put rows in each NewC shuffled order
    NewC = np.arange(df.shape[0])
    df = df.copy()  # TODO: check if this copy is really necessary

    unique_labels = list(pd.unique(labels))

    is1vector = len(unique_labels) == 2 and encode2as1vector

    matrix = _generate_y_PLSDA(labels, unique_labels, is1vector)

    if is1vector:
        # keep a copy to use later
        correct_labels = matrix.copy()

    # Use stratified n_fold cross-validation
    set_random = np.random.randint(1000000000)
    kf = StratifiedKFold(n_fold, shuffle=True, random_state=set_random)

    # Number of permutations + dataset with non-shuffled labels equal to iter_num + 1
    for i in range(iter_num + 1):
        # Temporary dataframe with rows in order of the NewC
        temp = df.iloc[NewC, :]
        # Setting up variables for results of the application of 3-fold cross-validated PLS-DA
        nright = 0

        # Repeating for each of the n groups
        for train_index, test_index in kf.split(df, labels):
            # plsda model building for each of the n stratified groups made
            plsda = PLSRegression(n_components=n_comp, scale=False)
            X_train, X_test = temp.iloc[train_index, :], temp.iloc[test_index, :]
            if not is1vector:
                y_train = matrix.iloc[train_index, :].copy()
                y_test = matrix.iloc[test_index, :].copy()

            else:
                y_train, y_test = correct_labels[train_index], correct_labels[test_index]
                correct = correct_labels[test_index]

            # Fitting the model
            plsda.fit(X=X_train, Y=y_train)

            # Predictions the test group
            y_pred = plsda.predict(X_test)

            # Decision rule for classification
            # Decision rule chosen: sample belongs to group where it has max y_pred (closer to 1)
            # In case of 1,0 encoding for two groups, round to nearest integer to compare
            # if not is1vector:
            #     for i in range(len(y_pred)):
            #         where_max = np.argmax(y_pred[i])

            if not is1vector:
                for i in range(len(y_pred)):
                    if list(y_test.iloc[i, :]).index(max(y_test.iloc[i, :])) == np.argmax(
                        y_pred[i]
                    ):
                        nright += 1  # Correct prediction
            else:
                rounded = np.round(y_pred)
                for i in range(len(y_pred)):
                    if rounded[i] == correct[i]:
                        nright += 1  # Correct prediction


        # Calculate accuracy for this iteration
        Accuracy.append(nright / len(labels))
        # Shuffle dataset rows, generating 1 permutation of the labels
        rng.shuffle(NewC)

    # Return also the K-fold cross-validation predictive accuracy for the non-shuffled dataset
    # and the p-value
    CV = Accuracy[0] # Predictive Accuracy of non-permuted dataset PLS-DA model - Accuracy[0]
    pvalue = (
        sum( [Accuracy[i] for i in range(1, len(Accuracy)) if Accuracy[i] >= Accuracy[0]] ) + 1
    ) / (iter_num + 1)

    return CV, Accuracy[1:], pvalue
