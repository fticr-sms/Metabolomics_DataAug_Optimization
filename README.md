# Metabolomics_DataAug_Optimization
Repository for the code with the optimization of GAN networks made during the project CPCA/A0/467905/2021. This code was made to be run in a Slurm Workload Manager job scheduler at the MACC's (Minho Advanced Computing Center) 'bob' cluster under the project name said above. As such, it will not be able to run on other platforms or a normal Python environment without some modifications.  Furthermore, it does not include the datasets (or auxiliary files) also needed to run the different scripts.

This repository include some of the optimizations of CWGAN-GP networks made for the paper: 'CWGAN-GP for data augmentation in supervised analysis of class imbalanced mass spectrometry metabolomics datasets.'

It does not have all the optimizations and includes optimizations for datasets that were not included in the paper as well as for feature occurrence data instead of just intensity data where the paper is focused. This is here to make available a great part of the code developed and ran under the project CPCA/A0/467905/2021.

We also thank the online platforms such as Tensorflow (as an example https://www.tensorflow.org/tutorials/generative/dcgan), Keras (as an example https://keras.io/examples/generative/wgan_gp/#wasserstein-gan-wgan-with-gradient-penalty-gp) and machinelearningmastery among others for their tutorials on the architecture and training of different types of GANs.
