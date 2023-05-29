#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:01:47 2021

@author: francoaquistapace

Copyright 2021 Franco Aquistapace

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
# Import modules and SOM
import pandas as pd
import time
import copy
import sys
from joblib import Parallel, delayed # For parallel processing


N_JOBS = int(sys.argv[1])

from SOM import *

# Import parameters
from input_params import PARAMS, LAYERS


# Fixed seed for consistent results
np.random.seed(1982) 


# Keep track of the header lines in each file
all_header_lines = []

# Keep track of the min and the max values of every feature 
# in the training data
all_min_max_features = []

# Let's streamline the process of opening a file and
# extracting the features
def open_and_extract(fname,features,training=False):
    '''
    Parameters
    ----------
    fname : str
        Sample file path.
    features : list of str
        List with the features of the sample that are going to be used.

    Returns
    -------
    Original and normalized dataframes with the selected features of the 
    sample, ready for training of the SOM or to be classified. Also returns 
    the header lines of the file.

    '''
    
    file = open(fname,'r')
    found_cols = False
    header_lines = []
    line_count = 0
    while not found_cols:
        line = file.readline()
        line_count += 1
        if 'ITEM: ATOMS' in line:
            columns = line.split()
            columns.pop(0) # Pop 'ITEM:'
            columns.pop(0) # Pop 'ATOMS'
            found_cols = True
        header_lines.append(line)
    file.close()
    
    df = pd.read_csv(fname,sep=' ', 
                     skiprows=line_count, names=columns)
    
    
    norm_df = df[features].copy()
    # For the training process we use the min and max values 
    # of every feature from the same sample, and save them to properly
    # rescale the samples to be analized later
    if training:
        for feat in features:
            min_value = norm_df[feat].min()
            max_value = norm_df[feat].max()
            delta = max_value - min_value
            norm_df[feat] = (norm_df[feat] - min_value) / delta
            
            # Save min and max for later
            min_max_features.append([min_value, max_value])
        
    return df, norm_df, header_lines



def normalize_data(df,features,scaling,min_max_features,training):
    '''

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the data to be normalized.
    features : list of str
        List containing the columns of the data to be used.
    scaling : list of str
        Must be of the same length as features. Either 'standard', 'robust' or 
        'normal' for standard scaling, robust scaling and linear 
        normalization, respectively.
    min_max_features : list of floats or ints
        List containing max and min values as tuples for every column 
        sepcified in features.
    training : bool
        Whether the data is for training or not. If True, then the max and min
        values of every column in features is obtained from the data in df and
        then stored in min_max_features. If False, the the max and min values 
        are fetched from min_max_features.

    Returns
    -------
    norm_df : pandas DataFrame
        DataFrame containing the normalized data from df[features].

    '''
    norm_df = df[features].copy()
    if training:
        for i in range(len(features)):
            feat = features[i]
            scale = scaling[i]
            if scale == 'normal':
                # Perform normal scaling
                min_value = norm_df[feat].min()
                max_value = norm_df[feat].max()
                delta = max_value - min_value
                if delta == 0:
                    delta = 1
                norm_df[feat] = (norm_df[feat] - min_value) / delta
                
                # Save min and max for later
                min_max_features.append([min_value, max_value])
                
            elif scale == 'standard':
                # Perform standard scaling
                mean = norm_df[feat].mean()
                std = norm_df[feat].std()
                if std == 0:
                    std = 1
                norm_df[feat] = (norm_df[feat] - mean)/std
                
                # Save mean, std, min and max for later
                min_max_features.append([mean, std])
                
            
            elif scale == 'robust':
                # Get 0.25 and 0.75 quantiles
                q25 = df[feat].describe()['25%']
                q75 = df[feat].describe()['75%']
                
                # Estimate IQR
                iqr = q75 - q25
                if iqr == 0:
                    iqr = 1
                # Get median
                median = df[feat].median()
                
                norm_df[feat] = (norm_df[feat] - median)/iqr
                
                # Save iqr and median
                min_max_features.append([iqr,median])
                
                
                
            else:
                print('Error: scaling values must be '+\
                      'either standard, robust or normal')
        
    else:
        for i in range(len(features)):
            feat = features[i]
            scale = scaling[i]
            if scale  == 'normal':
                min_value = min_max_features[i][0]
                max_value = min_max_features[i][1]
                delta = max_value - min_value
                
                norm_df[feat] = (norm_df[feat] - min_value) / delta
                
            elif scale == 'standard':
                mean = min_max_features[i][0]
                std = min_max_features[i][1]
                
                norm_df[feat] = (norm_df[feat] - mean)/std
                
            elif scale == 'robust':
                iqr = min_max_features[i][0]
                median = min_max_features[i][1]
                
                norm_df[feat] = (norm_df[feat] - median)/iqr
    
    return norm_df


# Auxiliary function that helps find missing items from list A in list B
def find_missing(A,B):
    '''

    Parameters
    ----------
    A : list
        Main list with the items we want to identify as missing in list B.
    B : list
        Secondary list, where we want to find the missing items.

    Returns
    -------
    New list, with the items in list A that are missing from list B.

    '''
    C = []
    for a in A:
        if not a in B:
            C.append(a)
    return C

# Auxiliary function to generate prime numbers
def generate_prime(n):
    primes = []
    m = 2
    while len(primes) < n:
        ADD = True
        for i in primes:
            if m % i == 0:
                ADD = False
        if ADD:
            primes.append(m)
        m += 1
    return primes[-1]


# We'll need an auxiliary function that maps the results of the secondary 
# layers of SOMs
def map_to_cluster_linear(groups_df, c_id, c_next_id, 
                          use_int=False, max_int=None, min_int=None,
                          N=1):
    '''s

    Parameters
    ----------
    groups_df : pandas Series
        Series containing the results of a secondary layer SOM prediction.
    c_id : int
        Id of the super cluster associated with the groups_df results.
    c_next_id : int
        Id of the next inmediate cluster.
    use_int : bool
        Whether or not the groups_df variable is actually an int value. When 
        True it allows to perform the mapping for a single value.
    max_int : int or None
        Maximum value to use when mapping a single int. It is only required 
        when use_int is True.
    min_int : int or None
        Minimum value to use when mapping a single int. It is only required 
        when use_int is True.
    N : int
        Amount of groups to use as (max_val - min_val) when this difference is
        zero. Default is 1.

    Returns
    -------
    Performs the following mapping on the given Series:
    new_cluster = c_id + 
                delta * lim * (groups_df.copy() - min_val)/(max_val - min_val)
        
    And returns a new Series with the mapping.

    '''
    if not use_int:
        max_val, min_val = groups_df.max(), groups_df.min()
        # Check that min and max vals are different
        if max_val - min_val == 0:
            delta = c_next_id - c_id
            lim = (N-1) / N
            mapped_df = c_id +\
                delta * lim * groups_df.copy() / (N-1)
        else:
            delta = c_next_id - c_id
            lim = (max_val/(max_val+1))
            mapped_df = c_id +\
                delta * lim * (groups_df.copy() - min_val)/(max_val - min_val)
        
        return mapped_df
    
    else:
        max_val, min_val = max_int, min_int
        delta = c_next_id - c_id
        lim = (max_val/(max_val+1))
        mapped_df = c_id +\
            delta * lim * (groups_df - min_val)/(max_val - min_val)
        
        return mapped_df
        

def map_to_cluster_godel(prev_id, layer, group):
    '''
    Parameters
    ----------
    prev_id : int
        Godel id of the super cluster that contains the new sub cluster.
    layer : int
        Layer of the new sub cluster, with layers starting at layer = 1.
    group : int
        Group number of the sub cluster, with groups starting at group = 1.

    Returns
    -------
    Godel mapping of the (layer,group) information of a new sub cluster
    based on the super cluster that contains it.

    '''
    # Get p_i
    p_i = generate_prime(layer)
    # Generate new id
    new_id = prev_id * (p_i ** group)
    return new_id


def map_to_cluster_default(groups_df):
    '''
    Parameters
    ----------
    groups_df : pandas Series
        Series containing the results of a secondary layer SOM prediction.
    
    Returns
    -------
    Returns the Series as it was given, this function is only used to allow
    for a default mapping that matches the methodology of the other mappings.
    '''
    mapped_df = groups_df
    return mapped_df
    

# Check that the input is well specified
PASS = True
# Training file
if PARAMS['training_file'] == '':
    print('Error: training_file must be specified')
    PASS = False
# Either file list or search pattern
if len(PARAMS['file_list']) == 0 and PARAMS['search_pattern'] == '':
    print('Error: either a list of files or a'+\
          ' search pattern must be specified')
    PASS = False
# Mapping must either be godel or linear
if not (PARAMS['mapping'] in ['godel','linear','default']):
    print('Error: mapping must either be \'godel\' or \'linear\'')
    PASS = False
# Check layer specs
for layer in LAYERS:
    # There must be at least 1 feature
    if len(layer['features']) == 0:
        print('Error: At least one feature must be specified for' +\
              ' each layer')
        PASS = False
    # scaling and features must be the same length
    if len(layer['features']) != len(layer['scaling']):
        print('Error: scaling and features must be the same length ' +\
              'for each layer')
        PASS = False
    # f must be such that 0 < f <= 1
    if layer['f'] > 1 or layer['f'] <= 0:
        print('Error: For every layer, f must be such that 0 < f <= 1')
        PASS = False
        
    # N must be an int
    if layer['N'] != int(layer['N']):
        print('Error: N must be an int for every layer')
        PASS = False
        
    # batched must either be True or False
    if not (layer['Batched'] == True or layer['Batched'] == False):
        print('Error: batched must be either True or False for every layer')
        PASS = False

if PASS == False:
    exit()
    
# Initialize mapping type
MAPPING = PARAMS['mapping']

# Initialize features
ALL_FEATURES = [LAYERS[i]['features'] for i in range(len(LAYERS))]
ALL_SCALES = [LAYERS[i]['scaling'] for i in range(len(LAYERS))]

print('Initializing SOM layers...')

# Build SOM model
SIGMAS = [LAYERS[i]['sigma'] for i in range(len(LAYERS))]
ETAS = [LAYERS[i]['eta'] for i in range(len(LAYERS))]
# Build sizes
SIZES = []
for i in range(len(LAYERS)):
    if len(ALL_FEATURES[i]) == 1:
        SIZES.append([2,LAYERS[i]['N']])
    else:
        SIZES.append([len(ALL_FEATURES[i]),LAYERS[i]['N']])

# Check whether the user wants to perform serial or batched training
BATCHED = [LAYERS[i]['Batched'] for i in range(len(LAYERS))]
BATCH_SIZES = [LAYERS[i]['Batch_size'] for i in range(len(LAYERS))]

# Initialize SOMs with first layer
soms = []
if BATCHED[0] == False:
    # Use serial SOM 
    soms.append([SOM(sigma=SIGMAS[0], eta=ETAS[0], size=SIZES[0])])
elif BATCHED[0] == True:
    # Use batched SOM
    soms.append([SOM_Batched(sigma=SIGMAS[0], eta=ETAS[0], 
                             size=SIZES[0], batch_size=BATCH_SIZES[0])])
else:
    print('Error: Batched must be either True or False')



# TRAINING 
# Start timing...
time1 = time.time()

# We're going to keep track of the clusters found in every layer
# during the training process
found_clusters = []

# Get training data and shuffle it
training_path = PARAMS['training_file']
print('Preparing training data from file %s' % training_path)
og_df = open_and_extract(training_path, [], training=False)[0]
min_max_temp = []
og_training_df = normalize_data(df=og_df, features=ALL_FEATURES[0], 
                                scaling=ALL_SCALES[0], 
                                min_max_features=min_max_temp, 
                                training=True)

# Save min_max features form first layer
all_min_max_features.append([min_max_temp.copy()])

fs = [LAYERS[i]['f'] for i in range(len(LAYERS))]
if fs[0] == '1':
    f = int(1)
else:
    f = float(fs[0])
training_df = og_training_df.sample(frac=f)


# Train the SOM
print('\nTraining SOM, layer 1...')
soms[0][0].train(training_df)


# --------------------------------
# SECONDARY LAYERS TRAINING
norm_df = normalize_data(df=og_df, features=ALL_FEATURES[0], 
                         scaling=ALL_SCALES[0], 
                         min_max_features=all_min_max_features[0][0], 
                         training=False)

results = soms[0][0].predict(norm_df)
# We only need the last column, which contains the grouping result
result_cols = results.columns.to_list()
groups = results[result_cols[-1]]
# Get unique groups
unique_groups = list(groups.unique())
# Start recording found clusters
found_clusters.append([unique_groups])
# Get initial mapped found clusters
mapped_found_clusters = copy.deepcopy(found_clusters)

if MAPPING == 'godel':
    mapped_groups = map_to_cluster_godel(prev_id=1, 
                                         layer=1, 
                                         group=groups+1)
    for j in range(len(found_clusters[0])):
        for k in range(len(found_clusters[0][j])):
            prev_id = 1
            mapped_found_clusters[0][j][k] = map_to_cluster_godel(
                                        prev_id=prev_id, layer=1, 
                                        group=k+1)

elif MAPPING == 'linear' or MAPPING == 'default':
    first_layer_mapping = {}
    for k in range(len(unique_groups)):
        first_layer_mapping[unique_groups[k]] = k
        mapped_found_clusters[0][0][k] = k
    mapped_groups = groups.map(first_layer_mapping)
    
# Concat new DataFrame
new_df = pd.concat([og_df,mapped_groups], axis=1)
col_list = new_df.columns.to_list()
new_df = new_df.rename(columns={col_list[-1]:'layer_1'})


# Get secondary layers results
for i in range(len(LAYERS)-1):
    print('\nTraining SOMs, layer ' + str(i+2) + '...')
    # Create an empy column for each secondary layer result
    layer_name = 'layer_' + str(i+2)
    new_df[layer_name] = 0
    
    # Initialize prevous layer data
    prev_layer_name = 'layer_' + str(i+1)
    
    # Save next layer found clusters
    layer_found_clusters = []
    
    # Now let's store every som in the next layer
    som_layer_next = []
    min_max_temp = []
    # Get corresponding super cluster in the previous layer
    for j in range(len(found_clusters[i])):
        
        print('Super SOM %d of %d:' % (j+1, len(found_clusters[i])))
        k_per_j = len(found_clusters[i][j])
        # Store the group SOMs
        som_group_next = []
        
        min_max_features_group_next = \
            [[] for k in range(len(found_clusters[i][j]))]
            
        # Check whether the user wants to perform serial or 
        # batched training, layer i + 1
        batched_next = BATCHED[i+1]
        if batched_next == False:
            for k in range(len(found_clusters[i][j])):
                # Use serial SOM 
                som_group_next.append(SOM(sigma=SIGMAS[i+1], 
                                          eta=ETAS[i+1], size=SIZES[i+1]))
        elif batched_next == True:
            batch_size = BATCH_SIZES[i+1]
            for k in range(len(found_clusters[i][j])):
                # Use batched SOM
                som_group_next.append(SOM_Batched(sigma=SIGMAS[i+1], 
                                                  eta=ETAS[i+1], 
                                                  size=SIZES[i+1], 
                                                  batch_size=batch_size))
        else:
            print('Error: Batched must be either True or False') 
            
        f_next = fs[i+1]
            
        for k in range(len(found_clusters[i][j])):
            print('Super cluster %d of %d' % (k+1, len(found_clusters[i][j])))
            # Get super cluster id
            cluster_id = mapped_found_clusters[i][j][k]
            # Only use the data associated with said super cluster
            cluster_data = new_df.loc[new_df[prev_layer_name] ==\
                                          cluster_id].copy()
            
            # Check if the data is empty
            if cluster_data.shape[0] == 0:
                continue
            
            som_id = int(k_per_j * j + k)
            cluster_data = normalize_data(cluster_data.copy(), 
                                ALL_FEATURES[i+1], 
                                ALL_SCALES[i+1],
                                min_max_features_group_next[k], 
                                training=True)
            
            
            if len(ALL_FEATURES[i+1]) == 1:
                if str(type(cluster_data)) ==\
                '<class \'pandas.core.series.Series\'>':
                    cluster_data = cluster_data.to_frame()
                cluster_data['aux_col'] = cluster_data
            
            training_data = cluster_data.sample(frac=f_next)
            
            # Train corresponding SOM
            som_group_next[k].train(training_data)
            
            # Predict results for the super cluster
            cluster_results = som_group_next[k].predict(cluster_data)

            # Select last column
            cluster_result_cols = cluster_results.columns.to_list()
            cluster_groups = cluster_results[cluster_result_cols[-1]]
            
            # Get unique and save to layer_found_clusters
            layer_found_clusters.append(list(cluster_groups.unique()))
            
            # Let's map the results to something that is 
            # easier to understand
            if MAPPING == 'godel':
                final_cluster_groups = map_to_cluster_godel(
                    prev_id = cluster_id, 
                    layer = i + 2, 
                    group = cluster_groups + 1)
                
            elif MAPPING == 'linear':
                j_cond = (j == len(found_clusters[i])-1)
                k_cond = (k == len(found_clusters[i][j])-1)
                if j_cond and k_cond:   
                    prev_cluster_id = mapped_found_clusters[i][j][k-1]
                    diff = abs(prev_cluster_id - cluster_id)
                    next_cluster_id = cluster_id + diff
                elif k_cond and not j_cond:
                    next_cluster_id = mapped_found_clusters[i][j+1][0]
                else:
                    next_cluster_id = mapped_found_clusters[i][j][k+1]
                    
                final_cluster_groups = map_to_cluster_linear(
                    groups_df = cluster_groups, 
                    c_id = cluster_id, 
                    c_next_id = next_cluster_id)
            
            elif MAPPING == 'default':
                final_cluster_groups = map_to_cluster_default(
                    cluster_groups)

            # Add results to final DataFrame
            new_df.loc[new_df[prev_layer_name] ==\
                cluster_id, [layer_name]] = final_cluster_groups
                
        # When done with the group, save SOMs and min_max features 
        # to the groups data
        som_layer_next.extend(som_group_next)
        min_max_temp.extend(min_max_features_group_next)
        
    # When done with the groups, save SOMs and min_max features to the layer 
    # data
    soms.append(som_layer_next)
    all_min_max_features.append(min_max_temp)
    
    # Save layer found clusters
    found_clusters.append(copy.deepcopy(layer_found_clusters))
    
    # Now let's get the mapped found clusters
    layer_mapped_found_clusters = copy.deepcopy(layer_found_clusters)
    if MAPPING == 'godel':
        for j in range(len(found_clusters[i+1])):
            for k in range(len(found_clusters[i+1][j])):
                # Get previous total super clusters
                prev_total = len(found_clusters[i+1])
                prev_classes = len(found_clusters[i])
                tot_per_class = int(prev_total / prev_classes)
                prev_id = mapped_found_clusters[i][
                    j//tot_per_class][j%tot_per_class]
                    
                layer_mapped_found_clusters[j][k] = map_to_cluster_godel(
                                        prev_id=prev_id, layer=i+2, 
                                        group=k+1)
                
    if MAPPING == 'linear':
        for j in range(len(found_clusters[i+1])):
            for k in range(len(found_clusters[i+1][j])):
                # Get previous total super clusters
                prev_total = len(found_clusters[i+1])
                prev_classes = len(found_clusters[i])
                tot_per_class = int(prev_total / prev_classes)
                cluster_id = mapped_found_clusters[i][
                    j//tot_per_class][j%tot_per_class]
                
                
                # Get inmediate next super cluster id
                j_cond = (j//tot_per_class == prev_classes - 1)
                k_cond = (j%tot_per_class == len(mapped_found_clusters[
                                                i][j//tot_per_class])-1)
                if j_cond and k_cond:
                    prev_cluster_id = mapped_found_clusters[i][ 
                        j//tot_per_class][j%tot_per_class - 1]
                    diff = abs(prev_cluster_id - cluster_id)
                    next_cluster_id = cluster_id + diff
                elif k_cond and not j_cond:
                    next_cluster_id = mapped_found_clusters[i][
                        j//tot_per_class + 1][0]
                else:
                    next_cluster_id = mapped_found_clusters[i][
                        j//tot_per_class][j%tot_per_class + 1]
                    
                # Get max_int and min_int
                max_int = max(found_clusters[i+1][j])
                min_int = min(found_clusters[i+1][j])
                
                layer_mapped_found_clusters[j][k] = map_to_cluster_linear(
                    groups_df=found_clusters[i+1][j][k],
                    c_id=cluster_id, 
                    c_next_id=next_cluster_id,
                    use_int=True,
                    max_int=max_int,
                    min_int=min_int)

    if MAPPING == 'default':
        for j in range(len(found_clusters[i+1])):
            for k in range(len(found_clusters[i+1][j])):
                layer_mapped_found_clusters[j][k] = map_to_cluster_default(
                    found_clusters[i+1][j][k])
                
    # Finally, add these results to the mapped_found_clusters
    mapped_found_clusters.append(copy.deepcopy(layer_mapped_found_clusters))
            

# --------------------------------



time_training = time.time() 
minutes_training = (time_training - time1)//60
seconds_training = (time_training - time1)%60
print('SOMs trained succesfully in %d minutes and %.f seconds' % \
      (minutes_training,seconds_training)) 


    
    
    
# PREDICT ATOM GROUPS AND WRITE OUTPUTS FOR EVERY FILE REQUESTED

files = PARAMS['file_list']
search_pattern = PARAMS['search_pattern']


# Check if the user has specified a search pattern instead
if len(files) == 0 and not search_pattern == '':
    print('This mode is not available yet.')
    exit()

# File list mode
if not len(files) == 0:
    print('\nUsing %d processes for classification.' % N_JOBS)
    # Functionalize file analysis
    print('\nAnalizing %d files...' % len(files))
    def classify_file(n):
        '''
        This function contains all the process required to 
        analyze the files. It was built simply to be able
        to parallelize the code.

        Params:
            n : int
                File index.
        '''
        file = files[n]
        df, norm_df, header_lines = open_and_extract(file, ALL_FEATURES[0])
        
        norm_df = normalize_data(df=df, features=ALL_FEATURES[0], 
                                 scaling=ALL_SCALES[0], 
                                 min_max_features=all_min_max_features[0][0], 
                                 training=False)
        
        results = soms[0][0].predict(norm_df)
        # We only need the last column, which contains the grouping result
        result_cols = results.columns.to_list()
        groups = results[result_cols[-1]]
        if MAPPING == 'godel':
            mapped_groups = map_to_cluster_godel(prev_id=1, 
                                                 layer=1, 
                                                 group=groups+1)
        elif MAPPING == 'linear' or MAPPING == 'default':
            mapped_groups = groups.map(first_layer_mapping)
            
        # Concat new DataFrame
        new_df = pd.concat([df,mapped_groups], axis=1)
        col_list = new_df.columns.to_list()
        new_df = new_df.rename(columns={col_list[-1]:'layer_1'})
        
        
        # Get secondary layers results
        for i in range(len(LAYERS)-1):
            # Create an empy column for each secondary layer result
            layer_name = 'layer_' + str(i+2)
            new_df[layer_name] = 0
            
            # Initialize prevous layer data
            prev_layer_name = 'layer_' + str(i+1)
            
            # Get all the unique super cluster ids of the previous layer
            #prev_cluster_ids = list(new_df[prev_layer_name].unique())
            
            # Get corresponding super cluster in the previous layer
            for j in range(len(found_clusters[i])):
                k_per_j = len(found_clusters[i][j])
                for k in range(len(found_clusters[i][j])):
                    cluster_id = mapped_found_clusters[i][j][k]
                    # Only use the data associated with said super cluster
                    cluster_data = new_df.loc[new_df[prev_layer_name] ==\
                                                  cluster_id].copy()
                    
                    # Check if the data is empty
                    if cluster_data.shape[0] == 0:
                        continue
                    
                    som_id = int(k_per_j * j + k)
                    cluster_data = normalize_data(cluster_data.copy(), 
                                        ALL_FEATURES[i+1], 
                                        ALL_SCALES[i+1],
                                        all_min_max_features[i+1][som_id], 
                                        training=False)
                    
                    
                    if len(ALL_FEATURES[i+1]) == 1:
                        if str(type(cluster_data)) ==\
                        '<class \'pandas.core.series.Series\'>':
                            cluster_data = cluster_data.to_frame()
                        cluster_data['aux_col'] = cluster_data
                    
                    
                    # Predict results for the super cluster
                    cluster_results = soms[i+1][som_id].predict(cluster_data)
        
                    # Select last column
                    cluster_result_cols = cluster_results.columns.to_list()
                    cluster_groups = cluster_results[cluster_result_cols[-1]]
                    
                    # Let's map the results to something that is 
                    # easier to understand
                    if MAPPING == 'godel':
                        final_cluster_groups = map_to_cluster_godel(
                            prev_id = cluster_id, 
                            layer = i + 2, 
                            group = cluster_groups + 1)
                        
                    elif MAPPING == 'linear':
                        j_cond = (j == len(found_clusters[i])-1)
                        k_cond = (k == len(found_clusters[i][j])-1)
                        if j_cond and k_cond:   
                            prev_cluster_id = mapped_found_clusters[i][j][k-1]
                            diff = abs(prev_cluster_id - cluster_id)
                            next_cluster_id = cluster_id + diff
                        elif k_cond and not j_cond:
                            next_cluster_id = mapped_found_clusters[i][j+1][0]
                        else:
                            next_cluster_id = mapped_found_clusters[i][j][k+1]
                            
                        final_cluster_groups = map_to_cluster_linear(
                            groups_df = cluster_groups, 
                            c_id = cluster_id, 
                            c_next_id = next_cluster_id,
                            N = len(found_clusters[i][j]))

                    elif MAPPING == 'default':
                        final_cluster_groups = map_to_cluster_default(
                            cluster_groups)
                    
                    # Add results to final DataFrame
                    new_df.loc[new_df[prev_layer_name] ==\
                        cluster_id, [layer_name]] = final_cluster_groups
                
            
            
        
        
        # Save new file with the results assigned to each atom
        new_path = 'SOM_' + file
        new_file = open(new_path, 'w')
        
        # Write the header of the file
        feat_line = ''
        for i in range(len(LAYERS)):
            new_layer_line = ' layer_' + str(i+1)
            feat_line += new_layer_line
        feat_line += '\n'
        for i in range(len(header_lines)):
            line = header_lines[i]
            if i == len(header_lines) - 1:
                line = line.replace('\n', feat_line)
            new_file.write(line)
        
        # Now let's write the new data
        final_string = new_df.to_csv(index=False, sep=' ', 
                                 float_format='%s', header=False)
        new_file.write(final_string)
        
            
        new_file.close()
    
    # Implement parallel process
    process = Parallel(n_jobs=N_JOBS)(delayed(classify_file)(n) for n in range(len(files)))
        
        

# Finish timing
time2 = time.time()
minutes_final = (time2 - time1)//60
seconds_final = (time2 - time1)%60
print('\nProcess completed')
print('Elapsed time: %d minutes and %.f seconds' % \
      (minutes_final,seconds_final)) 