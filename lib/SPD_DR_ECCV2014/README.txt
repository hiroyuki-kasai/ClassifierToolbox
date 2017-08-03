----------------------
Citation Details
----------------------
  
Please cite the following article when using this source code:
  
  M. Harandi, M. Salzmann, and R. Hartley.
  From Manifold to Manifold: Geometry-Aware Dimensionality Reduction for SPD Matrices.
  European Conference on Computer Vision (ECCV), 2014.
  
  @incollection{Harandi_ECCV_2014,
				title={From Manifold to Manifold: Geometry-Aware Dimensionality Reduction for SPD Matrices},
				author={Harandi, Mehrtash T. and Salzmann, Mathieu and Hartley, Richard},
				booktitle= {European Conference on Computer Vision (ECCV)},
				year={2014},
				isbn={978-3-319-10604-5},
				volume={8690},
				series={Lecture Notes in Computer Science},
				editor={Fleet, David and Pajdla, Tomas and Schiele, Bernt and Tuytelaars, Tinne},
				publisher={Springer International Publishing},
				pages={17-32},
	} 
  
  DOI: 10.1007/978-3-319-10605-2_2
  
You can obtain a copy of this article via:
http://dx.doi.org/10.1007/978-3-319-10605-2_2

   

----------------------
License
----------------------
  
The source code is provided without any warranty of fitness for any purpose.
You can redistribute it and/or modify it under the terms of the
GNU General Public License (GPL) as published by the Free Software Foundation,
either version 3 of the License or (at your option) any later version.
A copy of the GPL license is provided in the "GPL.txt" file.



----------------------
Instructions and Notes
----------------------

This code uses some functions from the manopt toolbox (http://www.manopt.org). For the ease of the user, we have incorporated the required functions from manopt into the package. As such, there is no need to install 
the manopt toolbox. However, appropriate citation to the manopt toolbox is encouraged.

	@article{manopt,
			 author  = {Nicolas Boumal and Bamdev Mishra and P.-A. Absil and Rodolphe Sepulchre},
			 title   = {{M}anopt, a {M}atlab Toolbox for Optimization on Manifolds},
			 journal = {Journal of Machine Learning Research},
			 year    = {2014},
			 volume  = {15},
			 pages   = {1455--1459},
			 url     = {http://www.manopt.org}
	}




To execute the code, run the runme.m file. The toy data contains training and test data for a multiclass problem. The covariance descriptors contain 10 genuine features and 20 noisy
features. Properly reducing the dimensionality will result in covariance descriptors with less noisy features and hence increases the classification accuracy. 





