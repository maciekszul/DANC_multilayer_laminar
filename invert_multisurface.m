function out_file=invert_multisurface(dataset_path, mat_file, subj_id, session_id, run_id, epo)

addpath('/home/mszul/git/DANC_spm12/spm12')
spm('defaults','eeg');
spm_jobman('initcfg');

set(0,'DefaultFigureVisible','off')

patch_size=5;
n_temp_modes=4;

subj_info=tdfread(fullfile(dataset_path,'raw/participants.tsv'));
s_idx=find(strcmp(cellstr(subj_info.subj_id),subj_id));
nas=subj_info.nas(s_idx,:);
lpa=subj_info.lpa(s_idx,:);
rpa=subj_info.rpa(s_idx,:);

% Where to put output data
data_dir=fullfile(dataset_path,'derivatives/processed',subj_id, session_id);
output_dir=fullfile(data_dir, 'inverse');
if exist(output_dir,'dir')~=7
    mkdir(output_dir);
end
subj_fs_dir=fullfile(dataset_path,'derivatives/processed',subj_id,'fs');
subj_surf_dir=fullfile(subj_fs_dir,'surf');

% Data file to load
%data_file=fullfile(output_dir, sprintf('fmspm_converted_autoreject-%s-%s-%s-%s-epo.mat', subj_id, session_id, run_id, epo));
data_file=mat_file

mri_fname=fullfile(dataset_path,'raw', subj_id, 'mri', 'headcast/t1w.nii');
    
invert_multisurface_results=[];
invert_multisurface_results.subj_id=subj_id;
invert_multisurface_results.session_id=session_id;
invert_multisurface_results.run_id=run_id;
invert_multisurface_results.epo=epo;
invert_multisurface_results.patch_size=patch_size;
invert_multisurface_results.n_temp_modes=n_temp_modes;

surf_fname=fullfile(subj_surf_dir,'multilayer.ds.link_vector.nodeep.gii');

invert_multisurface_results.surf_fname=surf_fname;
    
% Create smoothed meshes
[smoothkern]=spm_eeg_smoothmesh_multilayer_mm(surf_fname, patch_size, 11);

% Coregistered filename
[path,base,ext]=fileparts(data_file);
coreg_fname=fullfile(output_dir, sprintf('multilayer_%s.mat',base));
 
res_woi=[-Inf Inf];
if strcmp(epo,'motor')
    res_woi=[-150 -50];
end

% Run inversion
out_file=invert_ebb(data_file, coreg_fname, mri_fname, surf_fname,...
    nas, lpa, rpa, patch_size, n_temp_modes, [-Inf Inf], res_woi);
invert_multisurface_results.res_surf_fname=out_file;

% Load mesh results
%mesh_results=gifti(out_file);
D=spm_eeg_load(coreg_fname);
M=D.inv{1}.inverse.M;
U=D.inv{1}.inverse.U{1};
MU=M*U;
It   = D.inv{1}.inverse.It;

mu_fname=fullfile(output_dir, sprintf('multilayer_MU_%s.tsv',base));
dlmwrite(mu_fname, MU, '\t');
invert_multisurface_results.mu_fname=mu_fname;

it_fname=fullfile(output_dir, sprintf('multilayer_It_%s.tsv',base));
dlmwrite(it_fname, It, '\t');
invert_multisurface_results.it_fname=it_fname;

res_fname=fullfile(output_dir, sprintf('multilayer_res_%s.tsv',base));
%dlmwrite(res_fname, mesh_results.cdata(:), '\t');
invert_multisurface_results.res_fname=res_fname;
    
out_file=fullfile(output_dir, sprintf('invert_%s_multilayer_results.json',base));

fid = fopen(out_file,'w');
fwrite(fid, jsonencode(invert_multisurface_results)); 
fclose(fid); 



