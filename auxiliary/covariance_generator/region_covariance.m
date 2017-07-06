function [RCM1, RCM2, RCM3, RCM4, RCM5] = region_covariance(img, options)
% Region covariance matrix (GCM) algorithm
%
% Inputs:
%       img                 image data
%       options             options
% Output:
%       RCM1                region covariance matrix of size 7x7 (RCM1) 
%                           pixel location, intensity, 1st&2nd-order gradient
%       RCM2                region covariance matrix of size 7x7 (RCM2)
%                           pixel location, 1st&2nd-order gradient, and edge orientation
%       RCM3                region covariance matrix of size 6x6 (RCM3)
%                           pixel location, and 1st&2nd-order gradient
%       RCM4                region covariance matrix of size 8x8 (RCM4) with
%                           pixel location, intensity, 1st&2nd-order gradient, and 
%                           edge orientation
%       RCM5                region covariance matrix of size 5x5 (RCM5) with
%                           intensity, 1st&2nd-order gradient (not appear in the paper)
%
%
% References:
%       Yanwei Pang, Yuan Yuan, and Xuelong Li 
%       'Gabor-Based Region Covariance Matrices for Face Recognition'
%       IEEE Transactions on Circuits and Systems for Video Technology vol.18, no.7, 2008.
%
%
% Created by H.Kasai on June 23, 2017


    symm = @(X) .5*(X+X');

    % check correct number of arguments
    if (nargin ~= 2)        
        error('Please use the correct number of input arguments!')
    end
    
    % extract options
    if ~isfield(options, 'spd_projection')
        spd_projection = true;
    else
        spd_projection = options.spd_projection;
    end    

    % check if the input image is grayscale
    if size(img,3) == 3     
        warning('The input RGB image is converted to grayscale!')
        img = rgb2gray(img);
    end
  
    % obtain image size
    [ysize, xsize] = size(img);
    num_of_pixels = ysize * xsize;

    % prepare array
    Feature_Mat = zeros(8, num_of_pixels);
    
    
    % calculate first-order and second-order derivatives
    img = double(img);
    [Ix, Iy] = gradient(img);   % first order partials
    [Ixx, Ixy] = gradient(Ix);  % second order partials
    [Iyx, Iyy] = gradient(Iy);  % second order partials   

    
    % calculate 9-dimensional feature vectior for every pixels
    pixel_cnt = 0;
    for y = 1 : ysize
        for x = 1 : xsize 
            %fprintf('[%d %d] %d)\n', y, x, pixel_cnt);
            pixel_cnt = pixel_cnt + 1;

            Feature_Mat(1, pixel_cnt)   = x;  
            Feature_Mat(2, pixel_cnt)   = y;  
            Feature_Mat(3, pixel_cnt)   = img(y,x);  
            
            Feature_Mat(4, pixel_cnt)   = abs(Ix(y,x));
            Feature_Mat(5, pixel_cnt)   = abs(Iy(y,x));  
            
            Feature_Mat(6, pixel_cnt)   = abs(Ixx(y,x)); 
            Feature_Mat(7, pixel_cnt)   = abs(Iyy(y,x)); 

            theta = atan(abs(Iy(y,x))/abs(Ix(y,x)));
            Feature_Mat(8, pixel_cnt)   = theta; 

        end
    end


    % calculate covariance matrix
    Feature_Mat_type1 = Feature_Mat;
    Feature_Mat_type1(8,:) = [];            % remove orientation (theta)
    RCM1 = cov(Feature_Mat_type1');         % case for Eq.(5)
    
    Feature_Mat_type2 = Feature_Mat;
    Feature_Mat_type2(3,:) = [];            % remove intensity
    RCM2 = cov(Feature_Mat_type2');         % case for Eq.(6)
    
    Feature_Mat_type3 = Feature_Mat;
    Feature_Mat_type3(8,:) = [];            % remove edge orientation (theta)  
    Feature_Mat_type3(3,:) = [];            % remove intensity
    RCM3 = cov(Feature_Mat_type3');         % case for Eq.(8)       
    
    RCM4 = cov(Feature_Mat');               % case for Eq.(9)
    
    Feature_Mat_type5 = Feature_Mat;
    Feature_Mat_type5(8,:) = [];            % remove edge orientation (theta)        
    Feature_Mat_type5(1:2,:) = [];          % remove location 
    RCM5 = cov(Feature_Mat_type5');         % additional case


    % project onto SPD if needed
    if spd_projection
        RCM1 = symm(spd_project(RCM1));
        RCM2 = symm(spd_project(RCM2));
        RCM3 = symm(spd_project(RCM3));
        RCM4 = symm(spd_project(RCM4));
        RCM5 = symm(spd_project(RCM5));
    end
   
end

