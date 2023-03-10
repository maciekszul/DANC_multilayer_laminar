import numpy as np
import nibabel as nib
from sklearn import neighbors


def check_maj(list_to_check):
    list_len = len(list_to_check)
    majority = list_len//2 + 1
    if len(set(list_to_check[:majority])) == 1:
        return list_to_check[0]
    else:
        item, count = np.unique(list_to_check, return_counts=True)
        return item[np.argmax(count)]


def transform_atlas(paths_annot, paths_fsavg_sphere, paths_fsnat_sphere, pial_path, pial_ds_path, pial_nodeep_path, return_dict=False):
    '''
    Transform fsaverage *.annot file atlas to vertex labels in the native space downsampled layer.
    Returns:
        - original atlas colours per vertex in format ready for trimesh,
        - labels per fertex
        - (optional) label to colour dictionary
    '''
    # combining annotation files
    annot_data_lr = [nib.freesurfer.io.read_annot(i) for i in paths_annot]
    annots_lr = {i: [] for i in ["label_ix", "color", "label", ]}
    for ix in range(len(annot_data_lr)):
        annots_lr["label_ix"].append(annot_data_lr[ix][0])
        annots_lr["color"].append(annot_data_lr[ix][1])
        annots_lr["label"].append(np.array(annot_data_lr[ix][2]))
    
    fsavg_spheres = [nib.load(i).agg_data()[0] for i in paths_fsavg_sphere]
    fsnat_spheres = [nib.load(i).agg_data()[0] for i in paths_fsnat_sphere]
    
    # mapping atlas labels between fsavereage and fsnative spheres
    nat_annots = []
    for lr in range(len(fsavg_spheres)):
        tree = neighbors.KDTree(fsavg_spheres[lr], leaf_size=20)
        nat_dict_sub = {}
        for i in range(fsnat_spheres[lr].shape[0]):
            distance, index = tree.query([fsnat_spheres[lr][i]], k=5)
            distance = distance.flatten()
            index = index.flatten()
            label_indexes = annots_lr["label_ix"][lr][index].flatten()
            label_index = check_maj(label_indexes)
            label = annots_lr["label"][lr][label_index]
            nat_dict_sub[i] = [distance, index, label_indexes, label]
        nat_annots.append(nat_dict_sub)
    
    pial = nib.load(pial_path).agg_data()[0]
    pial_ds = nib.load(pial_ds_path).agg_data()[0]
    
    # mapping fsnative brain to downsampled brain
    pial_tree = neighbors.KDTree(pial, leaf_size=10)
    pial_2_ds_map = []
    for i in range(pial_ds.shape[0]):
        dist, pial_index = pial_tree.query([pial_ds[i]], k=1)
        pial_2_ds_map.append(pial_index)
    pial_2_ds_map = np.array(pial_2_ds_map).flatten()
    
    # concatenating the annotations
    annots_order = np.array(
        [nat_annots[0][i][3] for i in nat_annots[0].keys()] + 
        [nat_annots[1][i][3] for i in nat_annots[1].keys()]
    )
    
    # selecting the annotation for the downsampled pial
    nat_pial_annot = annots_order[pial_2_ds_map]
    
    # removing the DEEP structures
    nodeep = nib.load(pial_nodeep_path).agg_data()[0]
    tree = neighbors.KDTree(pial_ds, leaf_size=10)
    indices = [tree.query([nodeep[i]], k=1)[1].flatten()[0] for i in range(nodeep.shape[0])]
    nat_pial_annot = nat_pial_annot[indices]
    
    color = np.concatenate(annots_lr["color"])
    labels = np.concatenate(annots_lr["label"])
    
    lab_col_map = {lab: color[ix] for ix, lab in enumerate(labels)}
    mesh_colors = [lab_col_map[lab][:4].flatten() for lab in nat_pial_annot]
    # TRIMESH object mesh.visual.vertex_colors = mesh_colors 
    # only accepts list of 1x4 RGBA arrays 
    if return_dict:
        return mesh_colors, nat_pial_annot, lab_col_map
    else:
        return mesh_colors, nat_pial_annot


def fsavg_vals_to_native(values, fsavg_sphere_paths, fsnat_sphere_paths, pial_path, pial_ds_path, pial_ds_nodeep):
    """
    Transform values in fsaverage vertex order that contains values to vertex values in the native space downsampled.
    Returns:
        - values 
    """
    
    fsavg_spheres = [nib.load(i).agg_data()[0] for i in fsavg_sphere_paths]
    fsnat_spheres = [nib.load(i).agg_data()[0] for i in fsnat_sphere_paths]
    pial = nib.load(pial_path).agg_data()[0]
    pial_ds = nib.load(pial_ds_path).agg_data()[0]
    
    # values from fsaverage to fsnative
    fsnat_vx_values = []
    for lr in range(len(fsavg_spheres)):
        tree = neighbors.KDTree(fsavg_spheres[lr], leaf_size=20)
        vx_value = []
        for xyz_ix in range(fsnat_spheres[lr].shape[0]):
            dist, vx_index = tree.query([fsnat_spheres[lr][xyz_ix]], k=1)
            vx_value.append(values[lr][vx_index].flatten())
        fsnat_vx_values.append(np.array(vx_value))
    
    fsnat_vx_values = np.concatenate(fsnat_vx_values)
    
    # mapping fsnative brain to downsampled brain
    pial_tree = neighbors.KDTree(pial, leaf_size=10)
    pial_2_ds_map = []
    for i in range(pial_ds.shape[0]):
        dist, pial_index = pial_tree.query([pial_ds[i]], k=1)
        pial_2_ds_map.append(pial_index)
    pial_2_ds_map = np.array(pial_2_ds_map).flatten()
    
    # downsampled
    # but maybe mean of neighbouring vertices 
    fsnat_ds_vx_values = fsnat_vx_values[pial_2_ds_map]
    
    # removing deep structures
    nodeep = nib.load(pial_ds_nodeep).agg_data()[0]
    tree = neighbors.KDTree(pial_ds, leaf_size=10)
    indices = [tree.query([nodeep[i]], k=1)[1].flatten()[0] for i in range(nodeep.shape[0])]
    
    fsnat_ds_vx_values = fsnat_ds_vx_values[indices]
    
    return fsnat_ds_vx_values.flatten()