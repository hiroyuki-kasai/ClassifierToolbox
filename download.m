% download.m 

close all;
clear;
clc;

% local folder
local_folder_name = 'dataset';

% remote url, path, and filelist
site_url = 'https://dl.dropboxusercontent.com/';
site_path = 'u/869853/Github/FaceRecognitionToolbox/dataset/';
filename_array = {'ORL_Face_img.mat', 'ORL_Face_img_cov.mat', 'AR_Face_img_27x20.mat', 'AR_Face_img_60x43.mat','Brodatz_texture_img_small_set.mat'};
file_num = length(filename_array);

% download files
for i = 1 : file_num
    file_name = filename_array{i};
    filename_full = sprintf('%s%s%s', site_url, site_path, file_name);
    filename_full_local = sprintf('%s/%s', local_folder_name, file_name);
    
    fprintf('# Downlaoding [%d/%d]\n', i, file_num);
    fprintf('  * file:\t "%s"\n', file_name);
    fprintf('  * from:\t "%s%s"\n', site_url, site_path); 
    fprintf('  * into:\t "%s"\n', filename_full_local); 
    fprintf('  * ........ ');
    websave(filename_full_local, filename_full);
    fprintf('done\n\n');
end

fprintf('# %d files have been successfully downloaded.\n', file_num);
