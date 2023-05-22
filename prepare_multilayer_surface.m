function prepare_multilayer_surface(subject_dir, output_folder, n_surfaces, file_name)

addpath('/home/mszul/git/DANC_spm12/spm12', '-begin')
addpath('/home/mszul/git/MEGsurfer');
spm('defaults','eeg');
spm_jobman('initcfg');
ft_defaults();

subj_surf_dir=fullfile(subject_dir,'surf');

surface=file_name;
    
% Compute link vectors and save in pial surface
ds_fname=fullfile(output_folder,sprintf('%s.ds.gii', surface));
ds=gifti(ds_fname);

norm=compute_surface_normals(subj_surf_dir, 'pial', 'link_vector');
ds.normals=[];
for i=1:n_surfaces
    ds.normals=[ds.normals; norm];
end
ds_lv_fname=fullfile(output_folder, sprintf('%s.ds.link_vector.gii', surface));
save(ds,ds_lv_fname);

pial_ds_nodeep_fname=fullfile(output_folder,'pial.ds.link_vector.nodeep.gii');
pial_ds_nodeep=gifti(pial_ds_nodeep_fname);
pial_ds_fname=fullfile(output_folder,'pial.ds.gii');
pial_ds=gifti(pial_ds_fname);
mapping=knnsearch(pial_ds.vertices,pial_ds_nodeep.vertices);
verts_to_rem=setdiff([1:size(pial_ds.vertices,1)],mapping);    
disp(verts_to_rem)
n_verts_per_layer=size(ds.vertices,1)/n_surfaces;
offset=0;
r=[];
for i=1:n_surfaces
    len = length(verts_to_rem)
    res = verts_to_rem + offset
    r(end+1:end+len)=res;
    offset=offset+(n_verts_per_layer+1);
end
ds_final=remove_vertices(ds, r);
ds_lv_rm_fname=fullfile(output_folder,sprintf('%s.ds.link_vector.nodeep.gii',surface));
save(ds_final,ds_lv_rm_fname);