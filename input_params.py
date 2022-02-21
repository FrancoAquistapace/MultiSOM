#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 17:02:58 2021

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

# These dictionaries contain all the information and parameters
# that are going to be given to the algorithm to perform its task.

"""
    file_list : list
        List with the sample paths that are going to be classified
        by the SOM. If the list is empty, the program uses the
        search_pattern instead.
        
    search_pattern : str
        Search pattern for the files that are going to be 
        classified by the SOM. For example: 'feature_sample.*.config'
        
    training_file : str
        Path of the sample file used to train the SOM.
        
    mapping : str
        Must be either 'godel' or 'linear'. Godel mapping uses prime number
        multiplication to assign the cluster id based on the superior 
        clusters, while linear mapping defines the cluster id in between the 
        super cluster it belongs to and the inmediately next super cluster.
        
    features : list
        List containing the names of the features to be used.
        
    scaling : list
        List containing the names of the scaling process for each feature.
        
    f : int or float
        Fraction of the sample data to use when training the SOM.
        Must be in the range (0,1].
                              
    sigma : int or float
        Maximum value for the sigma(t) function, which gives the
        standard deviation of a Gaussian neighborhood as a 
        function of the current iteration step.
        
    eta : int or float
        Maximum learning rate for the eta(t) function, which gives
        the learning rate as a function of the current iteration 
        step.
        
    N : int
        Number of output neurons of the SOM network, i.e. number 
        of groups in which to classify the atoms of the sample.
        
    Batched : bool
        Whether to use batched learning or not.
        
    Batch_size : int
        Size of the batches used in the training process.
    
"""

PARAMS = {'file_list' : ['new_feat_dump.ensayo.2900000.config'], 
          'search_pattern' : '', 
          'training_file' : 'new_feat_dump.ensayo.2900000.config',
          'mapping' : 'linear'}


LAYERS = [{'features' : ['gr_coord', 'csp12', 'csp18'],
          'scaling' : ['normal','normal','normal'],
          'f' : 1,
          'sigma' : 0.5, 
          'eta' : 0.5, 
          'N' : 5, 
          'Batched' : True,
          'Batch_size' : 100},
          
          {'features' : ['v_svm','csp18'],
            'scaling' : ['robust','robust'],
            'f' : 1,
            'sigma' : 0.7,
            'eta' : 1,
            'N' : 2,
            'Batched' : True,
            'Batch_size' : 100}, 
          
          {'features' : ['gr_coord','csp18'],
            'scaling' : ['standard','robust'],
            'f' : 1,
            'sigma' : 0.7,
            'eta' : 1,
            'N' : 2,
            'Batched' : True,
            'Batch_size' : 10}]
