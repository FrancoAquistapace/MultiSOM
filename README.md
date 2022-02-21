# MultiSOM

## Description

Multilayer algorithm of Self Organising Maps (also known as Kohonen Networks) implemented in Python for clustering of atomistic samples through unsupervised learning. The program allows the user to select wich per-atom quantities to use for training and application of the network, this quantities must be specified in the LAMMPS input file that is being analysed. The algorithm also requires the user to introduce some of the networks parameters for each layer of SOMs:

- features: Per-atom features that are going to be used for the analysis.
- scaling: Individual scaling method for each feature.
- _f_: Fraction of the input data to be used when training the network, must be between 0 and 1.
- SIGMA: Maximum value of the _sigma_ function, present in the neighbourhood function.
- ETA: Maximum value of the _eta_ funtion, which acts as the learning rate of the network.
- N: Number of output neurons of the SOM, this is the number of groups the algorithm will use when classifying the atoms in the sample.
- Whether to use batched or serial learning for the training process.
- B: Batch size, in case the training is performed with batched learning.

A global parameter that can also be changed is the mapping of the results for every layer. There are currently two options, the 'godel' mapping uses prime number multiplication to encode the groups and subgroups an atom is classified into, while the 'linear' mapping does this by linear interpolation of a given layer based on the results obtained by the previous layer.

The input file/s must be inside the same folder as the `main.py` file. Furthermore, the input file passed to the algorithm must have the LAMMPS dump format, or at least have a line with the following format:

`ITEM: ATOMS id x y z feature_1 feature_2 ...`

To run the software, simply execute the following command in a terminal (from the folder that contains the files and with a python environment activated):

`python3 main.py`

The algorithm parameters are stored in the `input_params.py` file, and can be changed using any text editor.

Check the software report in the repository for more information.

## Dependencies:
This software is written in Python 3.8.8 and uses the following external libraries:
- NumPy 1.20.1
- Pandas 1.2.4

(Both packages come with the basic installation of Anaconda at: https://www.anaconda.com/)
