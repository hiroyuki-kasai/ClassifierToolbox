% Test file for Gabor_region_covariance() and region_covariance()

close all;
clear;
clc;


%% load image
img = imread('../images/peppers.png');
% resize
img_resize = imresize(img, [32 32]);
% show resized image
%imshow(img_resize)


%% test Gabor_region_covariance()
clear options;
orientation = 8;                % orientation
scale = 5;                      % scale
options.gw_display = true;      % display Gabor wavelet
options.spd_projection = true;  % project onto SPD
[GRCM1, GRCM2, GRCM3] = Gabor_region_covariance(img_resize, orientation, scale, options);


%% test region_covariance()
clear options;
options.spd_projection = true;  % project onto SPD
[RCM1, RCM2, RCM3, RCM4, RCM5] = region_covariance(img_resize, options);

