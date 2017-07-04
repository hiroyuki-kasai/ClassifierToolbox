% download.m 

close all;
clear;
clc;

% local folder
local_folder_name = 'dataset';

% remote url, path, and filelist
site_url = 'http://www.kasailab.com/';
site_path = 'public/github/FaceRecognitionToolbox/';
filename_array = {'ORL_Face_img.mat', 'ORL_Face_img_cov.mat'};

% download files
for i=1:length(filename_array)
    file_name = filename_array{i};
    filename_full = sprintf('%s%s%s', site_url, site_path, file_name);
    filename_full_local = sprintf('%s/%s', local_folder_name, file_name);
    
    fprintf('# Downlaoding "%s" from %s into %s ... ', file_name, site_url, filename_full_local);
    
    websave(filename_full_local, filename_full);
    fprintf('done\n');
end
