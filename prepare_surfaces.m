function prepare_surfaces(subject_dir, output_folder)

addpath('/home/mszul/git/DANC_spm12/spm12')
addpath('/home/mszul/git/MEGsurfer');
spm('defaults','eeg');
spm_jobman('initcfg');

subj_surf_dir=fullfile(subject_dir,'surf');

surfaces={'pial','white'};

for s_idx=1:length(surfaces)
    surface=surfaces{s_idx};
    
    % Compute link vectors and save in pial surface
    orig_fname=fullfile(subj_surf_dir,sprintf('%s.gii', surface));
    orig=gifti(orig_fname);
    ds_fname=fullfile(output_folder,sprintf('%s.ds.gii', surface));
    ds=gifti(ds_fname);
    norm=compute_surface_normals(subj_surf_dir, surface, 'link_vector');
    ds.normals=norm;
    ds_lv_fname=fullfile(output_folder, sprintf('%s.ds.link_vector.gii', surface));
    save(ds,ds_lv_fname);
    ds_final=remove_deep_vertices(subject_dir, output_folder, ds, orig, ds_lv_fname, orig_fname);
    ds_lv_rm_fname=fullfile(output_folder,sprintf('%s.ds.link_vector.nodeep.gii',surface));
    save(ds_final,ds_lv_rm_fname);

    mapping=knnsearch(ds.vertices,ds_final.vertices);
    verts_to_rem=setdiff([1:size(ds.vertices,1)],mapping);    

    inflated_surf_fname=fullfile(subj_surf_dir,sprintf('%s.ds.inflated.gii', surface));
    inflated_surf=gifti(inflated_surf_fname);
    inflated_surf_final=remove_vertices(inflated_surf, verts_to_rem);
    rm_fname=fullfile(output_folder,sprintf('%s.ds.inflated.nodeep.gii',surface));
    save(inflated_surf_final,rm_fname);
end
