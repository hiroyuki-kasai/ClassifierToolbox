% ========================================================================
% Fisher Discriminative Dictionary Learning (FDDL), Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% The code is for the paper:

% M. Yang, L. Zhang, X. Feng and D. Zhang, 
% ¡§Fisher Discrimination Dictionary Learning for Sparse Representation,¡¨ in ICCV 2011.

% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------

demo.m   Face recognition demo on AR database with 300-d Eigenface feature

utilier : folder of FDDL functions, including
Eigenface_f:  function of computing PCA Projection Matrix
FDDL:         main function of FDDL
FDDL_Class_Energy:   function of computing energy of certain class
FDDL_FDL_Energy:     function of computing energy of all classes
FDDL_Gradient_Comp:  function of computing coding model's gradient
FDDL_INIC:           function of initializing representation coef
FDDL_INID:           function of initializing dictionary
FDDL_SpaCoef:        function of computing coding coefficient
FDDL_UpdateDi:       function of updating dictioary
IPM_SC:              sparse coding function
soft:                soft threholding function

%-------------------------------------------------------------------------
Contact: {csmyang,cslzhang}@comp.polyu.edu.hk