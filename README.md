# FaceRecognitionToolbox : A Matlab toolbox for face recognition.

----------

Authors: [Hiroyuki Kasai](http://kasai.kasailab.com/)

Collaborator: Kohei Yoshikawa

Last page update: July 07, 2017

Latest library version: 1.0.1 (see Release notes for more info)

Introduction
----------
This package provides various tools for face recogntion, i.e., classification, applicaiton. 



List of algorithms
---------
- Basis
    - **PCA** (Principal component analysis)
        - M. Turk and A. Pentland, "[Eigenfaces for recognition](https://www.cs.ucsb.edu/~mturk/Papers/jcn.pdf)," J. Cognitive Neurosci," vol.3, no.1, pp.71-86, 1991.
        - See also [wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis).
    - **ICA** (Independent component analysis)
        - See [wikipedia](https://en.wikipedia.org/wiki/Independent_component_analysis).
    - **LDA** (Linear discriminant analysis)
        - P. N. Belhumeur, J. P. Hespanha, and D. I. Kriegman, "Eigenfaces vs. Fisherfaces: recognition using class specific linear projection," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.19, no.7, pp.711-720, 1997.
        - See also [wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis).
    - **SVM** (Support vector machine)
        - See [wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
        - Use Matlab built-in library (svmfitcsvm and predict).
- **LRC** variant
    - **LRC** (Linear regression classification)
        - I. Nassem, M. Bennamoun, "[Linear regression for face recognition](http://ieeexplore.ieee.org/document/5506092/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.32, no.11, 2010.
    - **LDRC** (Linear discriminant regression classificatoin)
        - S.-M. Huang and J.-F. Yang, "[Linear discriminant regression classification for face recognition](http://ieeexplore.ieee.org/document/6373697/)," IEEE Signal Processing Letters, vol.20, no.1, pp.91-94, 2013.
    - **LCDRC** (Linear collaborative discriminant regression classificatoin)
        - X. Qu, S. Kim, R. Cui and H. J. Kim, "[Linear collaborative discriminant regression classification for face recognition](http://www.sciencedirect.com/science/article/pii/S1047320315001297)," J. Visual Communication Image Represetation, vol.31, pp. 312-319, 2015.
- **CRC** (Collaborative representation based classification)
    - L. Zhanga, M. Yanga, and X. Feng, "[Sparse representation or collaborative representation: which helps face recognition?](http://dl.acm.org/citation.cfm?id=2356341)," Proceedings of the 2011 International Conference on Computer Vision (ICCV'11), pp. 471-478, 2011.
- LSR variant
    - **LSR** (Least squares regression)
    - **DERLR** (Discriminative elastic-net regularized linear regression)
        - Z. Zhang, Z. Lai, Y. Xu, L. Shao and G. S. Xie, "Discriminative elastic-net regularized linear regression," IEEE Transactions on Image Processing, vol.26, no.3, pp.1466-1481, 2017.
- Low-rank factorization based
    - **NMF** (Non-negative matrix factorization)
        - Please refer [NMFLibrary](https://github.com/hiroyuki-kasai/NMFLibrary).
    - **Robust PCA classifier**
        - Clasifier uses SRC;
        - Use [SparseGDLibrary](https://github.com/hiroyuki-kasai/SparseGDLibrary).
- **RCM** based
    - **RCM+kNN** (Region covariance matrix algorithm)
        - O. Tuzel, F. Porikli, and P. Meer "[Region covariance: a fast descriptor for detection and classification](https://link.springer.com/chapter/10.1007/11744047_45)," European Conference on Computer Vision (ECCV2006), pp.589-600, 2006.
    - **GRCM+kNN** (Gabor-wavelet-based region covariance matrix algorithm)
        - Y. Pang, Y. Yuan, and X. Li, "[Gabor-based Region covariance matrices for face recognition](http://ieeexplore.ieee.org/document/4498432/)," IEEE Transactions on Circuits and Systems for Video Technology vol.18, no.7, 2008.
- **SRC** variant
    - **SRC** (Sparse representation classifcation) 
        - J. Wright, A. Yang, A. Ganesh, S. Sastry, and Y. Ma, "Robust face recognition via sparse representation," IEEE Transaction on Pattern Analysis and Machine Intelligence, vol.31, no.2, pp.210-227, 2009).
    - **ESRC** (Extended Sparse representation classifcation)
        - W. Deng, J. Hu, and J. Guo, "Extended SRC: Undersampled face recognition via intraclass variant dictionary," IEEE Transation on Pattern Analysis Machine Intelligence, vol.34, no.9, pp.1864-1870, 2012.
- Dictionary learning based
    - **K-SVD**
        - M. Aharon, M. Elad, and A.M. Bruckstein, "[The K-SVD: An algorithm for designing of overcomplete dictionaries for sparse representation](http://ieeexplore.ieee.org/document/1710377/)", IEEE Trans. On Signal Processing, Vol.54, no.11, pp.4311-4322, November 2006.
    - **LC-KSVD** (Label Consistent K-SVD)
        - Z. Jiang, Z. Lin, L. S. Davis, "[Learning a discriminative dictionary for sparse coding via label consistent K-SVD](http://ieeexplore.ieee.org/abstract/document/5995354/)," IEEE Conference on Computer Vision and Pattern Recognition (CVPR2011), 2011.
        - Z. Jiang, Z. Lin, L. S. Davis, "[Label consistent K-SVD: learning A discriminative dictionary for recognition](http://ieeexplore.ieee.org/document/6516503/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.35, no.11, pp.2651-2664, 2013.


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

Second to do: download datasets and external libraries
----------------------------
Run `download` for downloading datasets and external libraries.
```Matlab
%% Run the downloading script
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

License
-------
- This toobox is **free**, **non-commercial** and **open** source.
- The code provided in this toobox should only be used for **academic/research purposes**.
- Third party files are included.
    - [OMPBox](http://www.cs.technion.ac.il/~ronrubin/Software/ompbox10.zip) is used for [OMP](https://en.wikipedia.org/wiki/Matching_pursuit) (orthogonal matching pursuit) algorithm.
    - [KSVDBox](http://www.cs.technion.ac.il/~ronrubin/Software/ksvdbox13.zip) is used for K-SVD algorithm.
    - [LC-KSVD](https://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html).
    - DERLR.
    - [JACOBI_EIGENVALUE](https://people.sc.fsu.edu/~jburkardt/m_src/jacobi_eigenvalue/jacobi_eigenvalue.html) is a MATLAB library which computes the eigenvalues and eigenvectors of a real symmetric matrix.
    - [NMFLibrary](https://github.com/hiroyuki-kasai/NMFLibrary) is for [NMF](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization).
    - [SparseGDLibrary](https://github.com/hiroyuki-kasai/SparseGDLibrary) is for [Robust PCA](https://en.wikipedia.org/wiki/Robust_principal_component_analysis) classifier.
<br />


Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.kasailab.com/) (email: kasai **at** is **dot** uec **dot** ac **dot** jp)

<br />

Release Notes
--------------
* Version 1.0.7 (July 06, 2017)
    - Add and modify many items.
* Version 1.0.0 (July 01, 2017)
    - Initial version.

