function [GRCM1, GRCM2, GRCM3] = Gabor_region_covariance(img, orientation_size, scale_size, options)
% Generate Gabor-wavelet-based region covariance matrix (GRCM) algorithm
%
% Inputs:
%       img                 image data
%       orientation_size    numner of orientations
%       scale_size          numner of scale
%       options             options
% Output:
%       GRCM1               Gabor-wavelet-based region covariance matrix (GRCM1)
%                           without pixel location annd intensity info. based on Eq.(15).
%       GRCM2               Gabor-wavelet-based region covariance matrix (GRCM2) 
%                           based on Eq.(16).
%       GRCM3               Gabor-wavelet-based region covariance matrix (GRCM3) 
%                           without pixel location info. based on Eq.(17).
%
% References:
%       Yanwei Pang, Yuan Yuan, and Xuelong Li 
%       'Gabor-Based Region Covariance Matrices for Face Recognition,'
%       IEEE Transactions on Circuits and Systems for Video Technology vol.18, no.7, 2008.
%
%
% Note that this code calls Gabor_wavelet_pixel() that is originally created by 
% Chai Zhi, and is modifyed by this code's authors. The original code can
% be found in https://www.mathworks.com/matlabcentral/fileexchange/20709-2d-gabor-wavelets.
%
%
% Created by H.Kasai and K.Yoshikawa on June 22, 2017
% Modified by H.Kasai and K.Yoshikawa on June 24, 2017


    symm = @(X) .5*(X+X');
    
    
    % check correct number of arguments
    if (nargin ~= 4)        
        error('Please use the correct number of input arguments!')
    end
    
    % extract options
    if ~isfield(options, 'spd_projection')
        spd_projection_flag = true;
    else
        spd_projection_flag = options.spd_projection;
    end
    
    if ~isfield(options, 'gw_display')
        gw_display_flag = false;
    else
        gw_display_flag = options.gw_display;
    end    

    % check if the input image is grayscale
    if size(img, 3) == 3     
        warning('The input RGB image is converted to grayscale!')
        img = rgb2gray(img);
    end
    
    % obtain image size
    [ysize, xsize] = size(img);
    if rem(ysize,2) || rem(xsize,2)
        error('The size (%d x %d) of image should be even.', ysize, xsize);
    end
    num_of_pixels = ysize * xsize;

    % store parameters from argments
    usize   = orientation_size;     % orientation
    vsize   = scale_size;           % scale
    uv_size = usize * vsize;
    
    % set paramters
    Kmax    = 5; % Kmax = pi/2;
    f       = sqrt(2);
    Delta   = 2 * pi;
    Delta2  = Delta * Delta;
    

    % prepare array
    GW_Mat = cell(ysize, xsize);
    Feature_Mat = zeros(uv_size+3, num_of_pixels);

    
    % calculate Gabor wavelets and features for every pixels
    pixel_cnt = 0;
    for m = -ysize/2 + 1 : ysize/2
        for n = -xsize/2 + 1 : xsize/2 
            pixel_cnt = pixel_cnt + 1;
            y = m+ysize/2;
            x = n+xsize/2;

            y_i = zeros(1, uv_size);
            
            % for all
            Feature_Mat(1, pixel_cnt)   = x;  
            Feature_Mat(2, pixel_cnt)   = y;  
            Feature_Mat(3, pixel_cnt)   = img(y,x);  

            for v = 0 : vsize-1
                for u = 1 : usize
                    varphi_uv = Gabor_wavelet_pixel(m, n, Kmax, f, u, v, Delta2);   % Eq.(12)
                    g_uv = abs(double(img(y,x)) * varphi_uv);                       % Eq.(14)
                    uv_index = u + v*usize;
                    y_i(1,uv_index) = varphi_uv;
                    Feature_Mat(3+uv_index, pixel_cnt)  = g_uv;                     % Eq.(16)
                end
            end

            GW_Mat{y,x} = y_i;
        end
    end


    % calculate covariance matrix in RCM
    Feature_Mat_type1 = Feature_Mat;
    Feature_Mat_type1(3,:) = [];            % remove pixel location info.
    GRCM1 = cov(Feature_Mat_type1');        % case for Eq.(15)
    
    GRCM2 = cov(Feature_Mat');              % case for Eq.(17)
    
    Feature_Mat_type3 = Feature_Mat;
    Feature_Mat_type3(1:3,:) = [];          % remove pixel location annd intensity info.
    GRCM3 = cov(Feature_Mat_type3');        % case for Eq.(16)  
    
    
    % project onto SPD if needed
    if spd_projection_flag
        GRCM1 = symm(spd_project(GRCM1));   %spd_project(GRCM1);
        GRCM2 = symm(spd_project(GRCM2));   %spd_project(GRCM2);
        GRCM3 = symm(spd_project(GRCM3));   %spd_project(GRCM3);  
    end
    
    
    % display the generated Gabor wavelets
    if gw_display_flag
        figure
        for v = 0 : vsize-1
            for u = 1 : usize
                GW_uv = zeros(ysize, xsize);
                for y = 1:ysize
                    for x = 1:xsize
                        feature_vec_xy = GW_Mat{y,x};
                        uv_index = u + v*usize;
                        GW_uv(y,x) = feature_vec_xy(1, uv_index);
                    end
                end

                % Show the real part of Gabor wavelets
                subplot(vsize, usize, v * usize + u),imshow ( real( GW_uv ) ,[]); 
            end
        end  
    end
end

