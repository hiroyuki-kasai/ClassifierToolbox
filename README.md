# ClassifierToolbox : A Matlab toolbox for classifier.

----------

Authors: [Hiroyuki Kasai](http://kasai.kasailab.com/)

Collaborator: Kohei Yoshikawa

Last page update: July 11, 2017

Latest library version: 1.0.4 (see Release notes for more info)

Introduction
----------
This package provides various tools for classification, e.g., image classification, face recogntion, and related applicaitons. 



List of algorithms
---------
- **Basis**
    - **PCA** (Principal component analysis)
        - M. Turk and A. Pentland, "[Eigenfaces for recognition](https://www.cs.ucsb.edu/~mturk/Papers/jcn.pdf)," J. Cognitive Neurosci," vol.3, no.1, pp.71-86, 1991.
        - See also [wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis).
    - **ICA** (Independent component analysis)
        - See [wikipedia](https://en.wikipedia.org/wiki/Independent_component_analysis).
    - **LDA** (Linear discriminant analysis)
        - P. N. Belhumeur, J. P. Hespanha, and D. I. Kriegman, "[Eigenfaces vs. Fisherfaces: recognition using class specific linear projection](http://ieeexplore.ieee.org/document/598228/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.19, no.7, pp.711-720, 1997.
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
- **LSR** variant
    - **LSR** (Least squares regression)
    - **DERLR** (Discriminative elastic-net regularized linear regression)
        - Z. Zhang, Z. Lai, Y. Xu, L. Shao and G. S. Xie, "[Discriminative elastic-net regularized linear regression](http://ieeexplore.ieee.org/document/7814255/)," IEEE Transactions on Image Processing, vol.26, no.3, pp.1466-1481, 2017.
- **Low-rank matrix factorization** based
    - **NMF** (Non-negative matrix factorization)
        - Please refer [NMFLibrary](https://github.com/hiroyuki-kasai/NMFLibrary).
    - **[Robust PCA](https://en.wikipedia.org/wiki/Robust_principal_component_analysis) classifier**
        - E. Candes, X. Li, Y. Ma, and J. Wright, "[Robust Principal Component Analysis?](http://perception.csl.illinois.edu/matrix-rank/Files/RPCA_JACM.pdf)," Journal of the ACM, vol.58, no.3, 2011.
        - Classifier uses SRC.
        - Use [SparseGDLibrary](https://github.com/hiroyuki-kasai/SparseGDLibrary).
- **RCM** based
    - **RCM+kNN** (Region covariance matrix algorithm)
        - O. Tuzel, F. Porikli, and P. Meer "[Region covariance: a fast descriptor for detection and classification](https://link.springer.com/chapter/10.1007/11744047_45)," European Conference on Computer Vision (ECCV2006), pp.589-600, 2006.
    - **GRCM+kNN** (Gabor-wavelet-based region covariance matrix algorithm)
        - Y. Pang, Y. Yuan, and X. Li, "[Gabor-based Region covariance matrices for face recognition](http://ieeexplore.ieee.org/document/4498432/)," IEEE Transactions on Circuits and Systems for Video Technology vol.18, no.7, 2008.
- **SRC** variant
    - **SRC** (Sparse representation based classifcation) 
        - J. Wright, A. Yang, A. Ganesh, S. Sastry, and Y. Ma, "[Robust face recognition via sparse representation](http://ieeexplore.ieee.org/document/4483511/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.31, no.2, pp.210-227, 2009.
    - **ESRC** (Extended sparse representation based classifcation)
        - W. Deng, J. Hu, and J. Guo, "[Extended SRC: Undersampled face recognition via intraclass variant dictionary](http://ieeexplore.ieee.org/document/6133293/)," IEEE Transation on Pattern Analysis Machine Intelligence, vol.34, no.9, pp.1864-1870, 2012.
    - **SSRC** (Superposed sparse representation based classifcation) 
        - W. Deng, J. Hu, and J. Guo, "[In defense of sparsity based face recognition](http://ieeexplore.ieee.org/document/6618902/)," IEEE Conference on Computer Vision and Pattern Recognition (CVPR2013), 2013.
    - **SRC-RLS**
        - M. Iliadis, L. Spinoulas, A. S. Berahas, H. Wang, and A. K. Katsaggelos, "[Sparse representation and least squares-based classification in face recognition](http://ieeexplore.ieee.org/document/6952144/)," Proceedings of the 22nd European Signal Processing Conference (EUSIPCO), 2014.
    - **SDR-SLR** (Sparse- and dense-hybrid representation and supervised low-rank) 
        - X. Jiang, and J. Lai, "[Sparse and dense hybrid representation via dictionary decomposition for face recognition](http://ieeexplore.ieee.org/document/6905839/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.37, no.5, pp.1067-1079, 2015.
- **Dictionary learning** based
    - **K-SVD**
        - M. Aharon, M. Elad, and A.M. Bruckstein, "[The K-SVD: An algorithm for designing of overcomplete dictionaries for sparse representation](http://ieeexplore.ieee.org/document/1710377/)", IEEE Transactions On Signal Processing, vol.54, no.11, pp.4311-4322, November 2006.
    - **LC-KSVD** (Label Consistent K-SVD)
        - Z. Jiang, Z. Lin, L. S. Davis, "[Learning a discriminative dictionary for sparse coding via label consistent K-SVD](http://ieeexplore.ieee.org/abstract/document/5995354/)," IEEE Conference on Computer Vision and Pattern Recognition (CVPR2011), 2011.
        - Z. Jiang, Z. Lin, L. S. Davis, "[Label consistent K-SVD: learning A discriminative dictionary for recognition](http://ieeexplore.ieee.org/document/6516503/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.35, no.11, pp.2651-2664, 2013.
    - **FDDL** (Fisher Discriminative Dictionary Learning)
        - M. Yang, L. Zhang, X. Feng, and D. Zhang, "[Fisher discrimination dictionary learning for sparse representation](http://ieeexplore.ieee.org/document/6126286/)," IEEE International Conference on Computer Vision (ICCV), 2011.
- **Geometry-aware**
    - **R-KSRC (Stein kernel)** (Riemannian kernelized sparse representation classification)
        - M. Harandi, R. Hartley, B. Lovell and C. Sanderson, "[Sparse coding on symmetric positive definite manifolds using bregman divergences](http://ieeexplore.ieee.org/document/7024121/)," IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2016.
        - M. Harandi, C. Sanderson, R. Hartley and B. Lovell, "[Sparse coding and dictionary learning for symmetric positive definite matrices: a kernel approach](https://drive.google.com/uc?export=download&id=0B9_PW9TCpxT0eW00U1FVd0xaSmM)," European Conference on Computer Vision (ECCV), 2012.
    - **R-KSRC (Log-Euclidean kernel)** (Riemannian kernelized sparse representation classification)
        - P. Li, Q. Wang, W. Zuo, and L. Zhang, "[Log-Euclidean kernels for sparse representation and dictionary learning](http://ieeexplore.ieee.org/document/6751309/)," IEEE International Conference on Computer Vision (ICCV), 2013.
        - S. Jayasumana, R. Hartley, M. Salzmann, H. Li, and M. Harandi, "[Kernel methods on the Riemannian manifold of symmetric positive definite matrices](http://ieeexplore.ieee.org/document/6618861/)," IEEE Conference on Computer Vision and Pattern Recognition (CVPR2013), 2013.
        - S. Jayasumana, R. Hartley, M. Salzmann, H. Li, and M. Harandi, "[Kernel methods on the Riemannian manifold with Gaussian RBF Kernels](http://ieeexplore.ieee.org/document/7063231/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.37, no.12, 2015.
    - [Reference] **R-KSRC (Deta-dependent kernel)** [Not included in this package]
        - Y. Wu, Y. Jia, P. Li, J. Zhang, and J. Yuan, "[Manifold kernel sparse representation of symmetric positive definite matrices and its applications](http://ieeexplore.ieee.org/document/7145428/)," IEEE Transactions on Image Processing, vol.24, no.11, 2015.

<br />

Folders and files
---------

<pre>
./                              - Top directory.
./README.md                     - This readme file.
./run_me_first.m                - The scipt that you need to run first.
./demo.m                        - Demonstration script to check and understand this package easily. 
|algorithm/                     - Algorithms for classifcations.
|auxiliary/                     - Some auxiliary tools for this project.
|demo_examples/                 - Some demonstration files.
|lib/                           - 3rd party tools.
|dataset/                       - Folder where datasets are stored.
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

- If your computer is behind a proxy server, please configure your Matlab setting. See [this](http://jp.mathworks.com/help/matlab/import_export/proxy.html?lang=en).

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

<br />

Third party tools, libraries, and packages. 
-------
- Third party files are included.
    - [OMPBox](http://www.cs.technion.ac.il/~ronrubin/Software/ompbox10.zip) is used for [OMP](https://en.wikipedia.org/wiki/Matching_pursuit) (orthogonal matching pursuit) algorithm.
    - [KSVDBox](http://www.cs.technion.ac.il/~ronrubin/Software/ksvdbox13.zip) is used for K-SVD algorithm.
    - [SPAMS](http://spams-devel.gforge.inria.fr/downloads.html) is used for various lasso problems.
    - [LC-KSVD](https://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html).
    - [FDDL](http://www4.comp.polyu.edu.hk/~cslzhang/code/FDDL.zip).
    - [RSR](https://drive.google.com/uc?export=download&id=0B9_PW9TCpxT0ZVpGRDNLX3NCbXc).
    - [Learning Discriminative Stein Kernel for SPD Matrices and Its Applications](https://github.com/seuzjj/DSK/archive/master.zip).
    - [SDR-SLR](http://www3.ntu.edu.sg/home/EXDJiang/CodesPAMI2015.zip).
    - [R-KSRC (Log-Euclidean kernel)](http://www4.comp.polyu.edu.hk/~cslzhang/LogEKernel_Project/ICCV_LogEKernel_Code.zip).
    - DERLR.
    - [JACOBI_EIGENVALUE](https://people.sc.fsu.edu/~jburkardt/m_src/jacobi_eigenvalue/jacobi_eigenvalue.html) is a MATLAB library which computes the eigenvalues and eigenvectors of a real symmetric matrix.
    - [NMFLibrary](https://github.com/hiroyuki-kasai/NMFLibrary) is for [NMF](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization).
    - [SparseGDLibrary](https://github.com/hiroyuki-kasai/SparseGDLibrary) is for [Robust PCA](https://en.wikipedia.org/wiki/Robust_principal_component_analysis) classifier.
- Note that please see the corresponding license for each.
<br />


Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.kasailab.com/) (email: kasai **at** is **dot** uec **dot** ac **dot** jp)

<br />

Release Notes
--------------
* Version 1.0.4 (July 11, 2017)
    - Add and modify SSRC etc. 
* Version 1.0.3 (July 10, 2017)
    - Add and modify SDR-SLR etc. 
* Version 1.0.2 (July 07, 2017)
    - Add and modify RSR, SVM etc. 
* Version 1.0.1 (July 06, 2017)
    - Add and modify many items.
* Version 1.0.0 (July 01, 2017)
    - Initial version.

