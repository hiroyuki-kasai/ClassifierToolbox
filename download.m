% download.m 

close all;
clear;
clc;


%% dataset
fprintf('###### Dataset downlowd ######\n\n');
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
    
    fprintf('# Dataset downlaoding [%d/%d]\n', i, file_num);
    fprintf('  * file:\t "%s"\n', file_name);
    fprintf('  * from:\t "%s%s"\n', site_url, site_path); 
    fprintf('  * into:\t "%s"\n', filename_full_local); 
    fprintf('  * ........ ');
    websave(filename_full_local, filename_full);
    fprintf('done\n\n');
end

fprintf('# %d dataset files have been successfully downloaded.\n\n\n', file_num);


%% libraries
fprintf('###### Library downlowd ######\n\n');

cd lib/;
url_array = {'https://github.com/hiroyuki-kasai/NMFLibrary/archive/master.zip'};
file_num = length(url_array);

% download files
for i = 1 : file_num
    file_url = url_array{i};
    
    fprintf('# Library downlaoding [%d/%d]\n', i, file_num);
    fprintf('  * file:\t "%s"\n', file_url);    
    fprintf('  * into:\t "%s"\n', 'download.zip'); 
    fprintf('  * ........ ');
    websave('download.zip', file_url);
    fprintf('done.\n');
    
    fprintf('# unzip download.zip ... ');
    unzip download.zip;
    fprintf('done.\n');
    
    fprintf('# delete download.zip ... ');
    delete('download.zip');
    fprintf('done.\n\n');
end

cd ..;


fprintf('# %d library files have been successfully downloaded and installed.\n', file_num);


