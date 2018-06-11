% download.m 

close all;
clear;
clc;


%% dataset
fprintf('###### Dataset downlowd ######\n\n');
% local folder
local_folder_name = 'dataset';


if ~exist('dataset', 'dir')   
    mkdir('dataset');
end

% remote url, path, and filelist
site_url = 'http://www.kasailab.com/';
site_path = 'public/github/ClassifierToolbox/dataset/';
filename_array = {'ORL_Face_img.mat', 'ORL_Face_img_cov.mat', 'AR_Face_img_27x20.mat', ...
                    'AR_Face_img_60x43.mat','Brodatz_texture_img_small_set.mat', ...
                    'USPS.mat', 'MNIST.mat', ...
                    'test_cov.mat'};
dataset_num = length(filename_array);

% download files
for i = 1 : dataset_num
    file_name = filename_array{i};
    filename_full = sprintf('%s%s%s', site_url, site_path, file_name);
    filename_full_local = sprintf('%s/%s', local_folder_name, file_name);
    
    fprintf('# Dataset downlaoding [%d/%d]\n', i, dataset_num);
    
    if ~exist(filename_full_local, 'file')  
        fprintf('  * file:\t "%s"\n', file_name);
        fprintf('  * from:\t "%s%s"\n', site_url, site_path); 
        fprintf('  * into:\t "%s"\n', filename_full_local); 
        fprintf('  * ........ ');
        websave(filename_full_local, filename_full);
        fprintf('done\n\n');
    else   
        fprintf('  * ........ %s already exists. Skip downloading.\n\n', file_name);
    end
end

fprintf('# %d dataset files have been successfully downloaded.\n\n\n', dataset_num);



%% libraries
fprintf('###### Library downlowd ######\n\n');

if ~exist('lib', 'dir')   
    mkdir('lib');
end

cd lib/;

lib_num = 4;
lib_cell = cell(1,lib_num); 


lib_struct.name = 'NMFLibrary-master';
lib_struct.url = 'https://github.com/hiroyuki-kasai/NMFLibrary/archive/master.zip';
lib_cell{1} = lib_struct;

lib_struct.name = 'SparseGDLibrary-master';
lib_struct.url = 'https://github.com/hiroyuki-kasai/SparseGDLibrary/archive/master.zip';
lib_cell{2} = lib_struct;

lib_struct.name = 'DSK-master';
lib_struct.url = 'https://github.com/seuzjj/DSK/archive/master.zip';
lib_cell{3} = lib_struct;

lib_struct.name = 'KMeans_SPD_Matrices';
lib_struct.url = 'https://jp.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/46343/versions/2/download/zip';
lib_cell{4} = lib_struct;



% download files
for i = 1 : lib_num
    file_name = lib_cell{i}.name;
    file_url =  lib_cell{i}.url;
    
    fprintf('# Library downlaoding [%d/%d]\n', i, lib_num);
    
    if ~exist(file_name, 'dir')     
        fprintf('  * file:\t "%s"\n', file_name);    
        fprintf('  * from:\t "%s"\n', file_url); 
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
    else
        fprintf('  * ........ %s already exists. Skip downloading.\n\n', file_name);
    end
end

cd ..;


fprintf('# %d library files have been successfully downloaded and installed.\n', lib_num);