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
    - See [wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis).
- **ICA** (Independent component analysis)
    - See [wikipedia](https://en.wikipedia.org/wiki/Independent_component_analysis).
- **LDA** (Linear discriminant analysis)
    - See [wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis).
- **LRC** (Linear regression classification)
    - I. Nassem, M. Bennamoun, "[Linear regression for face recognition](http://ieeexplore.ieee.org/document/5506092/)," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.32, no.11, 2010.
- **LCDRC** (Linear collaborative discriminant regression classificatoin)
    - X. Qu, S. Kim, R. Cui and H. J. Kim, "[Linear collaborative discriminant regression classification for face recognition](http://www.sciencedirect.com/science/article/pii/S1047320315001297)," J. Visual Communication Image Represetation, vol.31, pp. 312-319, 2015.
- **CRC** (Collaborative representation based classification)
    - Lei Zhanga, Meng Yanga, and Xiangchu Feng, "[Sparse Representation or Collaborative Representation: Which Helps Face Recognition?](http://dl.acm.org/citation.cfm?id=2356341)," Proceedings of the 2011 International Conference on Computer Vision (ICCV'11), pp. 471-478, 2011.
- **GRCM** (Gabor-wavelet-based region covariance matrix algorithm)
    - Yanwei Pang, Yuan Yuan, and Xuelong Li, "[Gabor-Based Region Covariance Matrices for Face Recognition](http://ieeexplore.ieee.org/document/4498432/)," IEEE Transactions on Circuits and Systems for Video Technology vol.18, no.7, 2008.
- **NMF** (Non-negative matrix factorization)
    - Please refer [NMFLibrary](https://github.com/hiroyuki-kasai/NMFLibrary).

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
                                 

First to do
----------------------------
Run `run_me_first` for path configurations. 
```Matlab
%% First run the setup script
run_me_first; 
```

Usage example: ORL face dateset demo 
----------------------------
Now, just execute `demo` for demonstration of this package.
```Matlab
%% Execute the demonstration script
demo; 
```



Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.kasailab.com/) (email: kasai **at** is **dot** uec **dot** ac **dot** jp)

Release Notes
--------------
* Version 1.0.0 (July 01, 2017)
    - Initial version.

