
# Python File to include functions related to the evaluation of GANs.

# Needed imports

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from elips import plot_confidence_ellipse
import itertools

import multianalysis as ma


def evaluation_coverage_density(real_data, gen_data, k=5, metric='Euclidean'):
    """Calculates the coverage and density of the generated data in regards to the real data.
    
       Metric described in https://arxiv.org/abs/2002.09797.
       
       real_data: numpy 2D array; array containing the original real data
       gen_data: numpy 2D array; array containing the generated data pretended to be evaluated
       k: scalar (default: 5); number of neighbors
       metric: str (default: 'Euclidean'); metric to calculate distances between real and generated samples. See 
    scipy.spatial.distance.cdist for available metrics.
       
       returns (scalar, scalar); (density, coverage)
    """
    #sphere_radius = {}
    sphere_radius = []
    # Matrix with distances between samples
    real_sample_dist = dist.squareform(dist.pdist(real_data, metric=metric))
    
    for sample in range(len(real_sample_dist)):
        real_sample_dist[sample].sort()
        # Save the distance to the k-th nearest neigbor as the sphere radius 
        # (index 0 is the distance of the sample to itself)
        #sphere_radius[sample] = real_sample_dist[sample, k]
        sphere_radius.append(real_sample_dist[sample, k])
    
    # Matrix with distances between generated and real samples
    # real samples in rows
    real_gen_sample_dist = dist.cdist(real_data, gen_data, metric=metric)
    #print(real_gen_sample_dist)
    
    # See if each value for each real sample is below or above the corresponding sphere radius
    real_gen_sample_dist_bool = real_gen_sample_dist < np.array(sphere_radius).reshape(len(sphere_radius),1)
    
    # Density Calculation
    ngen_neighbor = real_gen_sample_dist_bool.sum()
    density = ngen_neighbor/(len(gen_data)*k)
    
    # Coverage Calculation
    real_sample_covered = 0
    for real_sample in range(real_gen_sample_dist.shape[0]):
        if real_gen_sample_dist_bool[real_sample].sum() > 0:
            real_sample_covered = real_sample_covered + 1
    coverage = real_sample_covered / len(real_data)
    
    return density, coverage


def evaluation_coverage_density_all_k_at_once(real_data, gen_data, k_max=None, metric='Euclidean'):
    """Calculates the coverage and density of the generated data in regards to the real data from 1 to k_max neighbors.

       Metric described in https://arxiv.org/abs/2002.09797.

       real_data: numpy 2D array; array containing the original real data
       gen_data: numpy 2D array; array containing the generated data pretended to be evaluated
       k_max: scalar (default: None); maximum neighbors to consider; if None, k_max equals number of samples in gen_data
       metric: str (default: 'Euclidean'); metric to calculate distances between real and generated samples. See
    scipy.spatial.distance.cdist for available metrics.

       returns (list, list); (list of densities from 1 to k_max neighbors, list of coverages from 1 to k_max neighbors)
    """
    # To store the results
    density_list = []
    coverage_list = []

    # Matrix with distances between samples
    real_sample_dist = dist.squareform(dist.pdist(real_data, metric=metric))
    if k_max == None:
        k_max = len(real_sample_dist-1)

    # Matrix with distances between generated and real samples
    # real samples in rows
    real_gen_sample_dist = dist.cdist(real_data, gen_data, metric=metric)
    #print(real_gen_sample_dist)

    for k in range(1, k_max):
        sphere_radius = []
        for sample in range(len(real_sample_dist)):
            real_sample_dist[sample].sort()
            # Save the distance to the k-th nearest neigbor as the sphere radius
            # (index 0 is the distance of the sample to itself)
            #sphere_radius[sample] = real_sample_dist[sample, k]
            sphere_radius.append(real_sample_dist[sample, k])

        # See if each value for each real sample is below or above the corresponding sphere radius
        real_gen_sample_dist_bool = real_gen_sample_dist < np.array(sphere_radius).reshape(len(sphere_radius),1)

        # Density Calculation
        ngen_neighbor = real_gen_sample_dist_bool.sum()
        density = ngen_neighbor/(len(gen_data)*k)

        # Coverage Calculation
        real_sample_covered = 0
        for real_sample in range(real_gen_sample_dist.shape[0]):
            if real_gen_sample_dist_bool[real_sample].sum() > 0:
                real_sample_covered = real_sample_covered + 1
        coverage = real_sample_covered / len(real_data)

        # Store results
        density_list.append(density)
        coverage_list.append(coverage)

        #print(k)
    return density_list, coverage_list


def local_intrinsic_dimensionality_estimator_byMLE(data, k=5, metric='euclidean'):
    """Calculates Local Intrinsic Dimensionality of the data as estimated by the Maximum Likelihood Estimation method.
       Based on equation 5) of https://arxiv.org/pdf/1905.00643.pdf.
        
       data: 2D numpy array-like data.
       k: scalar (default: 5); number of neighbors to consider for LID estimation.
       metric: str (default: 'euclidean'); metric to calculate distances between real and generated samples. See
    scipy.spatial.distance.pdist for available metrics.
       
       returns: LID score.
    """
    
    if k > (len(data)-1):
        raise ValueError('k cannot be a higher value than n-1 (excluding itself) with n being the number of samples in data.')
    
    # Matrix with distances between samples
    sample_dist = dist.squareform(dist.pdist(data, metric=metric))
    sample_dist.sort() # Order 

    # Select only the k neighbors and exclude distance to itself and make distances logarithmic
    sample_dist_neighbors = np.log(sample_dist[:,1:k+1])

    rmax = sample_dist_neighbors[:,-1] # maximum distance within the neighborhood for each sample
    rsum = sample_dist_neighbors.sum(axis=1) # sum of the distance from the sample to all its k nearest neighbors
    # in X \ {x} for each sample

    LID = (rmax - rsum/k)**(-1) # Estimating LID

    return LID.mean() 


def local_intrinsic_dimensionality_estimator_byMLE_all_k_at_once(data, k_max=None, metric='euclidean'):
    """Calculates Local Intrinsic Dimensionality of the data (2 to k_max neighbors) as estimated by the MLE method.
       Based on equation 5) of https://arxiv.org/pdf/1905.00643.pdf.
       MLE - Maximum Likelihood Estimation

       data: 2D numpy array-like data.
       k_max: scalar (default: None); maximum neighbors to consider for LID estimation; if None, k_max equals number of samples.
       metric: str (default: 'euclidean'); metric to calculate distances between real and generated samples. See
    scipy.spatial.distance.pdist for available metrics.

       returns: list of LID score (from 2 to k_max neighbors).
    """
    LIDs = []

    if k_max == None:
        k_max = len(data)-1
    if k_max > (len(data)-1):
        raise ValueError('k_max cannot be a higher value than n-1 (excluding itself), n being the number of samples in data.')

    # Matrix with distances between samples
    sample_dist = dist.squareform(dist.pdist(data, metric=metric))
    sample_dist.sort() # Order

    for k in range(2, k_max):
        # Select only the k neighbors and exclude distance to itself and make distances logarithmic
        sample_dist_neighbors = np.log(sample_dist[:,1:k+1])

        rmax = sample_dist_neighbors[:,-1] # maximum distance within the neighborhood for each sample
        rsum = sample_dist_neighbors.sum(axis=1) # sum of the distance from the sample to all its k nearest neighbors
        # in X \ {x} for each sample
        #print(rmax, rsum/k)
        LID = (rmax - rsum/k)**(-1) # Estimating LID

        #print(LID)
        LIDs.append(LID.mean())

    return LIDs


def cross_LID_estimator_byMLE(real_data, gen_data, k=5, metric='euclidean'):
    """Calculates Cross Local Intrinsic Dimensionality as estimated by the Maximum Likelihood Estimation method of real
    data to generated data - CrossLID(Xr; Xg)
       Based on equation 5) and 6) of https://arxiv.org/pdf/1905.00643.pdf.

       That is, CrossLID is calculated based on the distance of each real sample to its k closest neighbors in the
    generated data.

       real_data, gen_data: 2D numpy array-like data; respectively the real and generated data.
       k: scalar (default: 5); number of neighbors to consider for CrossLID estimation.
       metric: str (default: 'euclidean'); metric to calculate distances between real and generated samples. See 
    scipy.spatial.distance.cdist for available metrics.

       returns: CrossLID score.
    """

    if k > len(gen_data):
        raise ValueError('k cannot be a higher value than the number of samples of gen_data.')

    real_gen_sample_dist = dist.cdist(real_data, gen_data, metric=metric)

    real_gen_sample_dist.sort() # Order

    # Select only the k neighbors and make distances logarithmic
    real_gen_sample_dist_neighbors = np.log(real_gen_sample_dist[:,:k])

    rmax = real_gen_sample_dist_neighbors[:,-1] # maximum distance within the neighborhood for each sample
    rsum = real_gen_sample_dist_neighbors.sum(axis=1) # sum of the distance from the sample to all its k nearest neighbors
    # in X \ {x} for each sample

    LID = (rmax - rsum/k)**(-1) # Estimating LID

    return LID.mean()


def cross_LID_estimator_byMLE_all_k_at_once(real_data, gen_data, k_max=None, metric='euclidean'):
    """Calculates Cross Local Intrinsic Dimensionality as estimated by the Maximum Likelihood Estimation method of real
    data to generated data - CrossLID(Xr; Xg) - from 2 to k_max neighbors.
       Based on equation 5) and 6) of https://arxiv.org/pdf/1905.00643.pdf.

       That is, CrossLID is calculated based on the distance of each real sample to its k closest neighbors in the
    generated data.

       real_data, gen_data: 2D numpy array-like data; respectively the real and generated data.
       k_max: scalar (default: None); maximum neighbors to consider for CrossLID estimation;
    if None, k_max equals number of samples.
       metric: str (default: 'euclidean'); metric to calculate distances between real and generated samples. See
    scipy.spatial.distance.cdist for available metrics.

       returns: list of CrossLID scores from 2 to k_max neighbors.
    """
    CrossLID = []
    if k_max == None:
        k_max = len(gen_data)
    if k_max > len(gen_data):
        raise ValueError('k cannot be a higher value than the number of samples of gen_data.')

    real_gen_sample_dist = dist.cdist(real_data, gen_data, metric=metric)

    real_gen_sample_dist.sort() # Order

    for k in range(2, k_max):
        # Select only the k neighbors and make distances logarithmic
        real_gen_sample_dist_neighbors = np.log(real_gen_sample_dist[:,:k])

        rmax = real_gen_sample_dist_neighbors[:,-1] # maximum distance within the neighborhood for each sample
        rsum = real_gen_sample_dist_neighbors.sum(axis=1) # sum of the distance from the sample to all its k nearest neighbors
        # in X \ {x} for each sample

        LID = (rmax - rsum/k)**(-1) # Estimating LID

        CrossLID.append(LID.mean())

    return CrossLID


def perform_HCA(df, labels, metric='euclidean', method='average'):
    "Performs Hierarchical Clustering Analysis of a data set with chosen linkage method and distance metric."

    distances = dist.pdist(df, metric=metric)

    # method is one of
    # ward, average, centroid, single, complete, weighted, median
    Z = hier.linkage(distances, method=method)

    # Correct First Cluster Percentage
    corr_1st_cluster = ma.correct_1stcluster_fraction(Z, labels)

    return {'Z': Z, 'distances': distances, 'correct 1st clustering': corr_1st_cluster}


def create_sample_correlations(df1, df2, method='pearson', return_pvalues=False):
    """Calculate the correlation between each pairwise combination of rows of one DataFrame and rows of another DataFrame.

       df1, df2: pandas DataFrame; data to calculate the correlations.
       method: str (default: 'pearson'); type of correlation calculated; accepted: 'pearson', 'spearman' or 'kendall'.
       return_pvalues: bool (default: False); If True, function also returns p-value matrix of the correlations.

       returns: pandas DataFrame with correlations between samples of df1 (in rows) and of df2 (in columns). Also returns
    corresponding p-value DataFrame if return_pvalues is true.
    """
    correlations = np.empty((len(df1), len(df2)))
    pvalues = np.empty((len(df1), len(df2)))

    if method == 'pearson':
        for sample1 in range(len(df1.index)):
            for sample2 in range(len(df2.index)):
                corr, pvalues_corr = stats.pearsonr(df1.iloc[sample1], df2.iloc[sample2])
                correlations[sample1, sample2] = corr
                if return_pvalues:
                    pvalues[sample1, sample2] = pvalues_corr

    elif method == 'spearman':
        for sample1 in range(len(df1.index)):
            for sample2 in range(len(df2.index)):
                corr, pvalues_corr = stats.spearmanr(df1.iloc[sample1], df2.iloc[sample2])
                correlations[sample1, sample2] = corr
                if return_pvalues:
                    pvalues[sample1, sample2] = pvalues_corr

    elif method == 'kendall':
        for sample1 in range(len(df1.index)):
            for sample2 in range(len(df2.index)):
                corr, pvalues_corr = stats.kendalltau(df1.iloc[sample1], df2.iloc[sample2])
                correlations[sample1, sample2] = corr
                if return_pvalues:
                    pvalues[sample1, sample2] = pvalues_corr

    else:
        raise ValueError('method not recognized. Must be one of "pearson", "spearman", "kendall".')

    correlation_df = pd.DataFrame(correlations, columns=df2.index, index=df1.index)
    if return_pvalues:
        pvalues_df = pd.DataFrame(pvalues, columns=df2.index, index=df1.index)
        return correlation_df, pvalues_df

    return correlation_df


def characterize_data(dataset, name='dataset', target=None):
    "Returns some basic characteristics about the dataset."

    n_samples, n_feats = dataset.shape

    if target:
        n_classes = len(np.unique(target))
        Samp_Class = len(target)/len(np.unique(target)) # Number of Sample per Class

    avg_feature_value = dataset.values.flatten().mean() # Mean value in the dataset
    max_feature_value = dataset.values.flatten().max() # Maximum value in the dataset
    min_feature_value = dataset.values.flatten().min() # Minimum value in the dataset
    std_feature_value = dataset.values.flatten().std() # Standard Deviation value in the dataset
    median_feature_value = np.median(dataset.values.flatten()) # Median value in the dataset

    if target:
        return {'Dataset': name,
                '# samples': n_samples,
                '# features': n_feats,
                'feature value average (std)': f'{avg_feature_value} ({std_feature_value})',
                'feature value ranges': f'({min_feature_value} - {max_feature_value})',
                'feature value median': median_feature_value,
                '# classes': n_classes,
                'samples / class': Samp_Class,
                } 
    else:
        return {'Dataset': name,
                '# samples': n_samples,
                '# features': n_feats,
                'Feature value average (std)': f'{avg_feature_value} ({std_feature_value})',
                'Feature value ranges': f'({min_feature_value} - {max_feature_value})',
                'Feature value median': median_feature_value,
                }


def characterize_binary_data(dataset, name='dataset', target=None):
    "Returns some basic characteristics about the dataset."

    n_samples, n_feats = dataset.shape

    if target:
        n_classes = len(np.unique(target))
        Samp_Class = len(target)/len(np.unique(target)) # Number of Sample per Class

    max_n_features = dataset.sum(axis=1).max() # Nº of features of sample with the most features
    min_n_features = dataset.sum(axis=1).min() # Nº of features of sample with the least features
    avg_n_features = round(dataset.sum(axis=1).mean(),2) # Average number of features of a sample
    std_n_features = round(dataset.sum(axis=1).std(),2) # Standard Deviation of the number of features per sample
    median_n_features = np.median(dataset.sum(axis=1)) # Median number of features of a sample

    min_n_samples = dataset.sum(axis=0).min() # Minimum nº of samples a feature appears
    max_n_samples = dataset.sum(axis=0).max() # Maximum nº of samples a feature appears
    avg_n_samples = round(dataset.sum(axis=0).mean(),2) # Average number of samples a feature appears in
    std_n_samples = round(dataset.sum(axis=0).std(),2) # Standard Deviation of the number of samples a feature appears in

    if target:
        return {'Dataset': name,
                '# samples': n_samples,
                '# features': n_feats,
                'average # of features per sample (std)': f'{avg_n_features} ({std_n_features})',
                '# features per sample ranges': f'({min_n_features} - {max_n_features})',
                '# features per sample median': median_n_features,
                'average # of times feature appear in sample (std)': f'{avg_n_samples} ({std_n_samples})',
                'min and max # of times a feature appears in a sample': f'{min_n_samples} - {max_n_samples}',
                '# classes': n_classes,
                'samples / class': Samp_Class,
                } 
    else:
        return {'Dataset': name,
                '# samples': n_samples,
                '# features': n_feats,
                'average # of features per sample (std)': f'{avg_n_features} ({std_n_features})',
                '# features per sample ranges': f'({min_n_features} - {max_n_features})',
                '# features per sample median': median_n_features,
                'average # of times feature appear in sample (std)': f'{avg_n_samples} ({std_n_samples})',
                'min and max # of times a feature appears in a sample': f'{min_n_samples} - {max_n_samples}',
                } 


def plot_PCA(principaldf, label_colors, components=(1,2), title="PCA", ax=None):
    "Plot the projection of samples in the 2 main components of a PCA model."

    if ax is None:
        ax = plt.gca()

    loc_c1, loc_c2 = [c - 1 for c in components]
    col_c1_name, col_c2_name = principaldf.columns[[loc_c1, loc_c2]]
    
    ax.set_xlabel(f'{col_c1_name}')
    ax.set_ylabel(f'{col_c2_name}')

    unique_labels = principaldf['Label'].unique()

    for lbl in unique_labels:
        subset = principaldf[principaldf['Label']==lbl]
        ax.scatter(subset[col_c1_name],
                   subset[col_c2_name],
                   s=50, color=label_colors[lbl], label=lbl)

    #ax.legend(framealpha=1)
    ax.set_title(title, fontsize=15)

def plot_ellipses_PCA(principaldf, label_colors, components=(1,2),ax=None, q=None, nstd=2):
    "Plot confidence ellipses of a class' samples based on their projection in the 2 main components of a PCA model."
    
    if ax is None:
        ax = plt.gca()
    
    loc_c1, loc_c2 = [c - 1 for c in components]
    points = principaldf.iloc[:, [loc_c1, loc_c2]]

    unique_labels = principaldf['Label'].unique()

    for lbl in unique_labels:
        subset_points = points[principaldf['Label']==lbl]
        plot_confidence_ellipse(subset_points, q, nstd, ax=ax, ec=label_colors[lbl], fc='none')

def pca_sample_projection(df, labels, pca_to_project, whiten=False, samp_number=None):
    
    loadings = pca_to_project.components_
    
    if whiten:
        pc_coord = np.matmul(df, loadings.T) * np.sqrt(samp_number-1)/ pca_to_project.singular_values_
    
    else:
        pc_coord = np.matmul(df, loadings.T)
        
    pc_coord['Label'] = labels
    
    return pc_coord


def plot_tSNE(embedded_df, labels, label_colors, components=(1,2), title="tSNE", ax=None):
    "Plot the projection of samples in the 2 main dimensions of a tSNE model."
    
    if ax is None:
        ax = plt.gca()

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    unique_labels = set(labels)

    for lbl in unique_labels:
        subset = embedded_df[[i==lbl for i in labels]]
        ax.scatter(subset[:,0],
                   subset[:,1],
                   s=50, color=label_colors[lbl], label=lbl)

    #ax.legend(framealpha=1)
    ax.set_title(title, fontsize=15)
    
def plot_ellipses_tSNE(embedded_df, labels, label_colors, components=(1,2),ax=None, q=None, nstd=2):
    "Plot confidence ellipses of a class' samples based on their projection in the 2 main dimensions of a tSNE model."
    
    if ax is None:
        ax = plt.gca()

    unique_labels = set(labels)

    for lbl in unique_labels:
        subset_points = embedded_df[[i==lbl for i in labels]]
        plot_confidence_ellipse(subset_points, q, nstd, ax=ax, ec=label_colors[lbl], fc='none')


def plot_PLS(principaldf, label_colors, components=(1,2), title="PCA", ax=None):
    "Plot the projection of samples in the 2 main components of a PLS-DA model."
    
    if ax is None:
        ax = plt.gca()
    
    loc_c1, loc_c2 = [c - 1 for c in components]
    col_c1_name, col_c2_name = principaldf.columns[[loc_c1, loc_c2]]

    ax.set_xlabel(f'{col_c1_name}')
    ax.set_ylabel(f'{col_c2_name}')

    unique_labels = principaldf['Label'].unique()

    for lbl in unique_labels:
        subset = principaldf[principaldf['Label']==lbl]
        ax.scatter(subset[col_c1_name],
                   subset[col_c2_name],
                   s=50, color=label_colors[lbl], label=lbl)

    ax.set_title(title, fontsize=15)

def plot_ellipses_PLS(principaldf, label_colors, components=(1,2),ax=None, q=None, nstd=2):
    "Plot the projection of samples in the 2 main components of a PLS-DA model."
    
    if ax is None:
        ax = plt.gca()
    
    loc_c1, loc_c2 = [c - 1 for c in components]
    points = principaldf.iloc[:, [loc_c1, loc_c2]]

    unique_labels = principaldf['Label'].unique()

    for lbl in unique_labels:
        subset_points = points[principaldf['Label']==lbl]
        plot_confidence_ellipse(subset_points, q, nstd, ax=ax, ec=label_colors[lbl], fc='none')


# Plot dendrograms
def color_list_to_matrix_and_cmap(colors, ind, axis=0):
        if any(issubclass(type(x), list) for x in colors):
            all_colors = set(itertools.chain(*colors))
            n = len(colors)
            m = len(colors[0])
        else:
            all_colors = set(colors)
            n = 1
            m = len(colors)
            colors = [colors]
        color_to_value = dict((col, i) for i, col in enumerate(all_colors))

        matrix = np.array([color_to_value[c]
                           for color in colors for c in color])

        matrix = matrix.reshape((n, m))
        matrix = matrix[:, ind]
        if axis == 0:
            # row-side:
            matrix = matrix.T

        cmap = mpl.colors.ListedColormap(all_colors)
        return matrix, cmap

def plot_dendogram(Z, leaf_names, label_colors, title='', ax=None, no_labels=False, labelsize=12, **kwargs):
    "Plot a dendrogram based on a linkage matrix Z with leaf names colored by their label/class."
    
    if ax is None:
        ax = plt.gca()
    hier.dendrogram(Z, labels=leaf_names, leaf_font_size=10, above_threshold_color='0.2', orientation='left',
                    ax=ax, **kwargs)
    #Coloring labels
    #ax.set_ylabel('Distance (AU)')
    ax.set_xlabel('Distance (AU)')
    ax.set_title(title, fontsize = 15)
    
    #ax.tick_params(axis='x', which='major', pad=12)
    ax.tick_params(axis='y', which='major', labelsize=labelsize, pad=12)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #xlbls = ax.get_xmajorticklabels()
    xlbls = ax.get_ymajorticklabels()
    rectimage = []
    for lbl in xlbls:
        col = label_colors[lbl.get_text()]
        lbl.set_color(col)
        #lbl.set_fontweight('bold')
        if no_labels:
            lbl.set_color('w')
        rectimage.append(col)

    cols, cmap = color_list_to_matrix_and_cmap(rectimage, range(len(rectimage)), axis=0)

    axins = inset_axes(ax, width="5%", height="100%",
                   bbox_to_anchor=(1, 0, 1, 1),
                   bbox_transform=ax.transAxes, loc=3, borderpad=0)

    axins.pcolor(cols, cmap=cmap, edgecolors='w', linewidths=1)
    axins.axis('off')