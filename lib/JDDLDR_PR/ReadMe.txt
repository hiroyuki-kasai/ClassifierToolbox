%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This code is for the paper of Zhizhao Feng, Meng Yang, Lei Zhang, Yan Liu, David Zhang,  Joint discriminative dimensionality reduction and dictionary learning
for face recognition. Pattern Recognition 46 (2013) 2134¨C2143.  

========================================================================
% Joint discriminative dimensionality reduction and dictionary learning (JDDLDR), Version 1.0
% Copyright(c) 2013  Meng YANG, Zhizhao Feng, Lei Zhang, Yan Liu, David Zhang
% All Rights Reserved.
%
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
An experiment on FRGC with 3 training samples per class is used as an example.

demo_FRGC.m         A demo to jointly learn dictionary and dimension-reduction projection matrix and to do classification.

utilies : folder of JDDLDR functions, including

JDDLDR.m     JDDLDR main function
JDDLDR_DCinit.m     Initialization of JDDLDR
JDDLDR_UDC.m        Updating of dictionary and coefficient matrix
JDDLDR_UP3.m        Updating of dimension-reduction matrix
Fun_CRC.m           Collaborative representation based classifier
%-------------------------------------------------------------------------
Contact: yangmengpolyu@gmail.com; {cslzhang}@comp.polyu.edu.hk

