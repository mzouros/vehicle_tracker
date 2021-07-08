# Project: Vehicle Tracker

This repo contains a single Jupyter Notebook (Python) for the Machine Learning in Multimedia Data Course at the MSc Program of AI 2020-2022, organized by NCSR Demokritos and University of Piraeus.
The purpose of this study is to collect audio signals from different static places near a road, extract useful information from these signals and implement different types of Machine and Deep Learning algorithms, in order to train them to recognize, given a new audio signal, how many vehicles have passed during the signal’s duration.

Instructors: Mr. Theodoros Giannakopoulos - [tygiannak](https://github.com/tyiannak)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Mr. Maglogiannis Ilias - [imaglo](https://github.com/imaglo)

## Notebook Sections

### Imports

* [matplotlib](https://github.com/matplotlib/matplotlib), [plotly](https://github.com/plotly) and [seaborn](https://github.com/mwaskom/seaborn) for Visualization

* [numpy](https://github.com/numpy/numpy) for Matrices

* [opencv](https://github.com/opencv/opencv) for Computer Vision

* [pydub](https://github.com/jiaaro/pydub) and [scipy](https://github.com/scipy/scipy) for .wav conversion and manipulation

* [librosa](https://github.com/librosa/librosa) for Music Analysis

* [sounddevice](https://github.com/spatialaudio/python-sounddevice), [scipy](https://github.com/scipy/scipy) for Audio Recording

* [sklearn](https://github.com/scikit-learn/scikit-learn) for Machine Learning

* [keras](https://github.com/keras-team/keras) for Deep Learning

### Data Collection

* Approximately over 200 audio signals were collected near various types of roads (one lane, two lanes, boulevard, highway, crossroads, etc), but only 172 were compliant and used for this study
* Vehicles recorded could be of any type (motorcycles, cars, vans, trucks, buses, etc.) and size, as long as they have an engine (eg. bicycles are not calcuated)
* The recording were made via our smartphones' microphones and had a duration that lasted just over 30 seconds
* Each sample was named in accordance to its label (numeric), indicating the vehicles that passed by, plus an incremental index indicating the number of the recording (eg. 21_recording137)
* Recording Example (original, transformed, augmented): https://github.com/mzouros/vehicle_tracker/tree/main/RecordingExample

### Data Preparation

* Data preparation took place on the [Audacity](https://www.audacityteam.org/) application and included:
  * Audio trimming to exactly 30 seconds
  * Noise reduction where possible (eg. reduce the volume of birds singing or people talking)
  * Stereo to Mono transformation 
  * Conversion to .wav

### Algorithmic Approaches

* Machine Learning:
  * SVC
* Deep Learning:
  * CNN
  * LSTM
* In all three approaches, we extracted the labels from the audio signals names and we created 8 labels corresponding to approximately +5 vehicles each time. So our labels were ranging from 1-5 (label 0) to 35+ (label 7) vehicles detected.

### Feature Extraction and Data Augmentation

* Feauture extraction and augmentation was different for each of our three approaches:
  * Mel Spectograms for the CNN model. Augmentation via filtering/masking our Mel Spectograms on both their axes (mel scale, time)
  * MFCCs for the LSTM model. Augmentation via new audio signal generation, using various sound augmentation techniques (White Noise, Time and Pitch Shifting)
  * Time (Root Mean Square Energy, Zero Crossing Rate) and Frequency (Spectral Centroid, Spectral Bandwidth) Domain Features for the SVC algorithm. Augmentation via new audio signal generation, using Pitch Shifting (up, down)

#### Original VS White Noise Added VS Time Shifted VS Pitch Shifted Soundwave

![alt text](https://i.imgur.com/c7tPeiH.png)
![alt text](https://i.imgur.com/VfbtbW3.png)
![alt text](https://i.imgur.com/G56OQtq.png)
![alt text](https://i.imgur.com/AdjbuiQ.png)

#### Mel VS Masked Mel Spectogram

![alt text](https://i.imgur.com/3MIsDql.png)
![alt text](https://i.imgur.com/AoiugzB.png)

### Implementation and Results

* Tried with different kernels, regularization parameters (C) and kernel’s coefficients (gamma) for our SVC algorithm
* Tried with different architectures, model sizes (layers, nodes), batch sizes, kernel & stride sizes (CNN) and number of epochs for our NN models
* Tried most of overfitting avoidance techniques for our Deep Learning algorithms (kernel regularization, batch normalization, dropout, early stopping)

#### CNN Model
![alt text](https://i.imgur.com/WmGcHu2.png)

#### CNN Accuracy & Loss (30 epochs)
![alt text](https://i.imgur.com/d8weLVM.png)![alt text](https://i.imgur.com/h8D8r9f.png)

#### LSTM Model
![alt text](https://i.imgur.com/nIirdjT.png)

#### LSTM Accuracy & Loss (35 epochs)
![alt text](https://i.imgur.com/sA0QPny.png)![alt text](https://i.imgur.com/ldgXyJD.png)

#### SVC Model Complexity
![alt text](https://i.imgur.com/ZcZsaMM.png)

#### SVC Confusion Matrix
![alt text](https://i.imgur.com/8LQKtWA.png)

#### SVC Predictions
![alt text](https://i.imgur.com/Q9oxqMO.png)

### Discussion

* The results suggest that our two NN models don’t perform as well as our SVC algorithm
* The LSTM model seems to perform much better than the CNN during training, but still faces problems during the prediction stage
* Our SVC algorithm seems to have the best prediction results
* Prediction seems way harder in samples with lots of vehicles passing by
* Small and noisy datasets are better to be approached via the traditional machine learning techniques and algorithms

### Future Work

The study can be extended in a number of different ways:
* Better recording devices with noise reduction, different types of roads
* It can be implemented using a Regression approach
* It can be extended from vehicle detection to vehicle detection and classification
* A combination of both acoustic and image/video data could yield far more better prediction results

## Authors

* **Michael Zouros** - [mzouros](https://github.com/mzouros)
* **Evangelia Baou** - [LiaBaou](https://github.com/LiaBaou)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Theodoros Giannakopoulos - [tygiannak](https://github.com/tyiannak)
* Ilias Maglogiannis - [imaglo](https://github.com/imaglo)

