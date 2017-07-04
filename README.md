# FaceRecognitionToolbox : A Matlab toolbox for face recognition.

----------

Authors: [Hiroyuki Kasai](http://kasai.kasailab.com/) and Kohei Yoshikawa

Last page update: July 01, 2017

Latest library version: 1.0.0 (see Release notes for more info)

Introduction
----------
This package provides various tools for face recogntion, i.e., classification, applicaiton. 



List of benchmarks
---------
- **PCA** (Principal component analysis)
    - M. Turk and A. Pentland, "Eigenfaces for recognition," J. Cognitive Neurosci," vol.3, no.1, pp.71-86, 1991.
    - See [wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis).
- **ICA** (Independent component analysis)
    - See [wikipedia](https://en.wikipedia.org/wiki/Independent_component_analysis).
- **LDA** (Linear discriminant analysis)
    - P. N. Belhumeur, J. P. Hespanha, and D. I. Kriegman, "Eigenfaces versus Fisherfaces," Pattern Anal. Mach. Intell., vol.19, no. 7, pp.771-720, 1997.
    - See [wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis).
- **LRC** (Linear regression classification)
    - I. Nassem, M. Bennamoun, "[Linear regression for face recognition](http://ieeexplore.ieee.org/document/5506092/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.32, no.11, 2010.
- **LDRC** (Linear discriminant regression classificatoin)
    - S.-M. Huang and J.-F. Yang, "[Linear discriminant regression classification for face recognition](http://ieeexplore.ieee.org/document/6373697/)," IEEE Signal Processing Letters, vol.20, no.1, pp.91-94, 2013.
- **LCDRC** (Linear collaborative discriminant regression classificatoin)
    - X. Qu, S. Kim, R. Cui and H. J. Kim, "[Linear collaborative discriminant regression classification for face recognition](http://www.sciencedirect.com/science/article/pii/S1047320315001297)," J. Visual Communication Image Represetation, vol.31, pp. 312-319, 2015.
- **CRC** (Collaborative representation based classification)
    - Lei Zhanga, Meng Yanga, and Xiangchu Feng, "[Sparse Representation or Collaborative Representation: Which Helps Face Recognition?](http://dl.acm.org/citation.cfm?id=2356341)," Proceedings of the 2011 International Conference on Computer Vision (ICCV'11), pp. 471-478, 2011.
- **RCM+kNN** (Region covariance matrix algorithm)
    - O. Tuzel, F. Porikli, and P. Meer "[Region covariance: a fast descriptor for detection and classification](https://link.springer.com/chapter/10.1007/11744047_45)," European Conference on Computer Vision (ECCV2006), pp.589-600, 2006.
- **GRCM+kNN** (Gabor-wavelet-based region covariance matrix algorithm)
    - Yanwei Pang, Yuan Yuan, and Xuelong Li, "[Gabor-Based Region Covariance Matrices for Face Recognition](http://ieeexplore.ieee.org/document/4498432/)," IEEE Transactions on Circuits and Systems for Video Technology vol.18, no.7, 2008.
- **LSR** (Least squares regression)
- **DERLR** (Discriminative elastic-net regularized linear regression)
    - Z. Zhang, Z. Lai, Y. Xu, L. Shao and G. S. Xie, ?gDiscriminative Elastic-Net Regularized Linear Regression?h, IEEE Transactions on Image Processing, vol.26, no.3, pp.1466-1481, 2017.
- **NMF** (Non-negative matrix factorization)
    - Please refer [NMFLibrary](https://github.com/hiroyuki-kasai/NMFLibrary).

<br />

Folders and files
---------

<pre>
./                              - Top directory.
./README.md                     - This readme file.
./run_me_first.m                - The scipt that you need to run first.
./demo.m                        - Demonstration script to check and understand this package easily. 
./test_comparison_syntheric.m   - Demonstration script for synthetic dataset. 
./test_classification           - Demonstration script for real dataset. 
|auxiliary/                     - Some auxiliary tools for this project.
|covariance_generator/          - Tools for generating covariance descriptors.
|3rd_parth/                     - 3rd party tools.
</pre>
   

<br />

First to do: configure path
----------------------------
Run `run_me_first` for path configurations. 
```Matlab
%% First run the setup script
run_me_first; 
```                              

<br />

Second to do: download datasets
----------------------------
Run `download` for downloading datasets.
```Matlab
%% Run the downloading script for downloading datasets
download; 
```

- If your computer is behind a proxiy server, please configure your Matlab setting. See [this](http://jp.mathworks.com/help/matlab/import_export/proxy.html?lang=en).

<br />

Usage example: ORL face dateset demo: 3 steps!
----------------------------
Now, just execute `demo` for demonstration of this package.
```Matlab
%% Execute the demonstration script
demo; 
```

<br />

The "**demo.m**" file contains below.
```Matlab
%% load data
load('./dataset/ORL_Face_img_cov.mat');

%% perform RCM k-NN classifier with 
% GRCM2 with eigenvalue-based distance
grcm_accuracy = rcm_knn_classifier(TrainSet, TestSet,'GRCM', '2', 'EV', 5);
% RCM4 with eigenvalue-based distance
rcm_accuracy = rcm_knn_classifier(TrainSet, TestSet, 'RCM', '4', 'EV', 5);

%% show recognition accuracy
fprintf('# GRCM2 Accuracy = %5.2f\n', grcm_accuracy);
fprintf('# RCM4 Accuracy = %5.2f\n', rcm_accuracy);
```

<br />

Let take a closer look at the code above bit by bit. The procedure has only **3 steps**!

**Step 1: Load data**

First, we load datasets including train set and test set. This case uses a covariance dataset that is originally generated from [ORL face dataset](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).
```Matlab    
load('./dataset/ORL_Face_img_cov.mat');
```

**Step 2: Perform solver**

Now, you can perform optimization solvers, i.e., RCM-based [kNN classifier](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm), calling `rcm_knn_classifier()` function with appropriate paramters. 
```Matlab
% GRCM2 with eigenvalue-based distance
grcm_accuracy = rcm_knn_classifier(TrainSet, TestSet, 'GRCM', '2', 'EV', 5);

% RCM4 with eigenvalue-based distance
rcm_accuracy = rcm_knn_classifier(TrainSet, TestSet, 'RCM', '4', 'EV', 5);
```
The first case performs the Gabor-wavelet-based region covariance matrix (CRCM) algorithm (type 4) with eigen-value based disctance followed by 5-NN classifier. 
The second cases peforms the standard region covariance matrix (RCM) algorithm (type 2) with the same setting as before. They return the final accuracy.

**Step 3: Show recognition accuracy**

Finally, the final recognition accuracis are shown.
```Matlab
fprintf('# GRCM2 Accuracy = %5.2f\n', grcm_accuracy);
fprintf('# RCM4 Accuracy = %5.2f\n', rcm_accuracy);
```

That's it!

<br />


Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.kasailab.com/) (email: kasai **at** is **dot** uec **dot** ac **dot** jp)

<br />

Release Notes
--------------
* Version 1.0.0 (July 01, 2017)
    - Initial version.

