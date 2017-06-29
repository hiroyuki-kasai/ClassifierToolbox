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
- PCA
- LDA

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

