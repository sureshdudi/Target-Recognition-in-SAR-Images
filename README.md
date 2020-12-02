# Target-Recognition-in-SAR-Images

# Problem Statement                                                                                                                                                             
SAR(Synthetic aperture radar) is an imaging radar that transmits microwaves which generates imagery through the reflected microwaves from the objects in both high and azimuth range resolutions. The good things about SAR is that it works in all weather conditions, day/night so it has many application areas like navigation, guidance, remote sensing, reconnaissance, resource exploration etc.. It is difficult to recognize an object in SAR imagery due to absence of colour information and shape reflection from a target changes. So, here the problem statement is to recognize the targets automatically in SAR have always been a challenge in research community.
# Background                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
In past so many years, many solutions have been proposed for SAR target recognition which are based on the 2 steps i.e feature extraction and classification. Many algorithms like Principal component analysis, independent component analysis, Hu invariants moments, sparse matrix, non-negative matrix etc... have been used to extract SAR target feature extraction. For classification purpose, algorithms like hidden markov models, artificial neural network(ANN), support vector machine (SVM), Adaboost, mean square error etc.. can be used. To improve performance of such systems, more good classifier along with better mechanism for extracting discriminative features is needed.
Deep learning area in recent years is getting much attention is recent years as it has the capability to learn feature automatically from the raw data and at the same time performs discrimination for more accuracy. CNN is one of the deep learning architecture has been used successfully in image recognition along with convolutional auto-encoder(CAE) can be used for SAR target recognition. Here CAE is used for generating the feature vector for an image. This is done by the reconstruction of the image through its representation along with minimization of error between reconstructed and input image. It can extract features from SAR images automatically and classify them in different classes.
# Methodology
Architecture of the target recognition in SAR images based on deep learning is shown in figure 3.
#### Step 1: Data collection and dataset preparation
The dataset for this work is MSTAR which includes military vehicles i.e. ten tank targets. Pre-processing is done based on the requirements.
#### Step 2: Dataset is divided into training dataset for creating the model and test dataset for testing the dataset.
#### Step 3: Developing a model based on Convolutional neural network and convolutional auto-encoder for target recognition in SAR images.
#### Step 4: Training and experimentation on datasets
#### step 5: Testing the model on real time data


# Dataset
The dataset that can be used for this work is MSTAR (Moving and Stationary Target Acquisition                                                                                                                                                                           
and Recognition).
