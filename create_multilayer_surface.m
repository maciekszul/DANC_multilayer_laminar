function create_multilayer_surface(subject_dir, layers_n, output_dir, name_prefix, singularity_path, mount_dir, varargin)

addpath('/home/mszul/git/DANC_spm12/spm12');
addpath('/home/mszul/git/MEGsurfer');

hemispheres={'lh','rh'};

output_name = strcat(name_prefix, '.ds.gii');
layers = linspace(0, 1, layers_n);
layers = layers(2:end-1);


parpool(length(layers))

parfor l=1:length(layers)
    layer=layers(l);
    for h=1:length(hemispheres)
        hemi=hemispheres{h};
        wm_file=fullfile(subject_dir, 'surf', sprintf('%s.white',hemi));
        out_file=fullfile(output_dir, sprintf('%s.%.1f',hemi,layer));
        [status, out]=system(sprintf('singularity exec -B %s %s mris_expand -thickness %s %d %s', mount_dir, singularity_path, wm_file, layer, out_file))
    end
end


% Read RAS offset from freesurfer volume
ras_off_file =fullfile(subject_dir, 'mri', 'orig.mgz');
[status, out]=system(sprintf('singularity exec -B %s %s mri_info --cras %s', mount_dir, singularity_path, ras_off_file));
cols=strsplit(out,' ')
ras_offset=[str2num(cols{1}) str2num(cols{2}) str2num(cols{3})];

% Convert freesurfer surface files to gifti
for l=1:length(layers)
    layer=layers(l);
    for h_idx=1:length(hemispheres)    
        hemi=hemispheres{h_idx};
        orig_name=fullfile(output_dir, sprintf('%s.%.1f', hemi, layer));
        new_name=fullfile(output_dir, sprintf('%s.%.1f.gii', hemi, layer));
        system(sprintf('singularity exec -B %s %s mris_convert %s %s', mount_dir, singularity_path, orig_name, new_name));
   
        % Read in each hemisphere's gifti file and adjust for RAS offset
        g=gifti(new_name);
        % Set transformation matrix to identiy
        g.mat=eye(4);
        g=set_mat(g,'NIFTI_XFORM_UNKNOWN','NIFTI_XFORM_TALAIRACH');
        % Apply RAS offset
        g.vertices=g.vertices+repmat(ras_offset,size(g.vertices,1),1);
        save(g, new_name);
    end
    
    % combine hemispheres
    lh=fullfile(output_dir, sprintf('lh.%.1f.gii', layer));
    rh=fullfile(output_dir, sprintf('rh.%.1f.gii', layer));
    combined=fullfile(output_dir, sprintf('%.1f.gii', layer));
    combine_surfaces({lh, rh}, combined);
end

% downsample
in_surfs={fullfile(subject_dir, 'surf', 'white.gii')};
out_surfs={fullfile(output_dir, 'white.ds.gii')};
for l=1:length(layers)
    layer=layers(l);
    in_surfs{end+1}=fullfile(output_dir, sprintf('%.1f.gii', layer));
    out_surfs{end+1}=fullfile(output_dir, sprintf('%.1f.ds.gii', layer));
end

in_surfs{end+1}=fullfile(subject_dir, 'surf', 'pial.gii');
out_surfs{end+1}=fullfile(output_dir, 'pial.ds.gii');

decimate_multiple_surfaces(in_surfs, out_surfs, 0.1);

combined_name=fullfile(output_dir, output_name);
% reverse order so surface order matches electrode order in laminar recordings
out_surfs(end:-1:1) = out_surfs(:);
combine_surfaces(out_surfs, combined_name);
