function prepare_multilayer_surface(dataset_path, subj_id, n_surfaces, an_name)

addpath('/home/bonaiuto/spm12')
addpath('/home/bonaiuto/MEGsurfer');
spm('defaults','eeg');
spm_jobman('initcfg');

subj_fs_dir=fullfile(dataset_path,'derivatives/processed',subj_id,'fs');
subj_surf_dir=fullfile(subj_fs_dir,'surf');

surface=an_name;
    
% Compute link vectors and save in pial surface
ds_fname=fullfile(subj_surf_dir,sprintf('%s.ds.gii', surface));
ds=gifti(ds_fname);

norm=compute_surface_normals(subj_surf_dir, 'pial', 'link_vector');
ds.normals=[];
for i=1:n_surfaces
    ds.normals=[ds.normals; norm];
end
ds_lv_fname=fullfile(subj_surf_dir, sprintf('%s.ds.link_vector.gii', surface));
save(ds,ds_lv_fname);

pial_ds_nodeep_fname=fullfile(subj_surf_dir,'pial.ds.link_vector.nodeep.gii');
pial_ds_nodeep=gifti(pial_ds_nodeep_fname);
pial_ds_fname=fullfile(subj_surf_dir,'pial.ds.gii');
pial_ds=gifti(pial_ds_fname);
mapping=knnsearch(pial_ds.vertices,pial_ds_nodeep.vertices);
verts_to_rem=setdiff([1:size(pial_ds.vertices,1)],mapping);    

n_verts_per_layer=size(ds.vertices,1)/n_surfaces;
offset=0;
r=[];
for i=n_surfaces:-1:1
    r(end+1:end+length(verts_to_rem))=verts_to_rem+offset;
    offset=offset+(n_verts_per_layer+1);
end
ds_final=remove_vertices(ds, r);
ds_lv_rm_fname=fullfile(subj_surf_dir,sprintf('%s.ds.link_vector.nodeep.gii',surface));
save(ds_final,ds_lv_rm_fname);

