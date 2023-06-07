import numpy as np
import nibabel as nib
from sklearn import neighbors
from scipy.ndimage import gaussian_filter
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


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

def compute_csd(surf_tcs, times, mean_dist, n_surfs):
    # Compute CSD
    nd = 1
    spacing = mean_dist*10**-3

    csd=np.zeros((n_surfs, surf_tcs.shape[1]))
    for t in range(surf_tcs.shape[1]):
        phi=surf_tcs[:,t]
        csd[0,t]=surf_tcs[0,t]
        csd[1,t]=surf_tcs[1,t]
        for z in range(2,n_surfs-3):
            csd[z,t]=(phi[z+2]-2*phi[z]+phi[z-2])/((nd*spacing)**2)
        csd[-2,t]=surf_tcs[-2,t]
        csd[-1,t]=surf_tcs[-1,t]            
    
    return csd


def smooth_csd(csd, n_surfs):
    # interpolate CSD in space
    y = np.linspace(0,n_surfs-1,n_surfs)
    Yi=np.linspace(0,n_surfs-1,500)
    
    f=interp1d(y,csd,kind='cubic',axis=0)
    csd_smooth=f(Yi)
    
    csd_smooth=savgol_filter(csd_smooth, 51, 3, axis=1)
    
    return csd_smooth


def compute_rel_power(power, freqs):
    power = gaussian_filter(power, sigma=[1.5, 2])

    if np.min(power[:]) < 0:
        power = power - np.min(power[:])
    rel_power = np.zeros(power.shape)
    for freq in range(len(freqs)):
        rel_power[:, freq] = (power[:, freq] - np.min(power[:, freq])) / (
                    np.max(power[:, freq]) - np.min(power[:, freq]))

    return rel_power


def get_crossover(freqs,rel_per_power,rel_aper_power):
    n_chans=rel_per_power.shape[0]
    ab_idx = np.where((freqs >= 7) & (freqs <= 30))[0]
    g_idx = np.where((freqs >= 50) & (freqs <= 125))[0]
    ab_rel_pow = np.mean(rel_per_power[:, ab_idx], axis=1)
    g_rel_pow = np.mean(rel_aper_power[:, g_idx], axis=1)
    crossovers = detect_crossing_points(ab_rel_pow, g_rel_pow)
    assert(len(crossovers)<=2)
    if len(crossovers) > 1:
        dist1 = np.min([crossovers[0], n_chans - crossovers[0]])
        dist2 = np.min([crossovers[1], n_chans - crossovers[1]])
        if dist1 > dist2:
            crossover = crossovers[0]
        else:
            crossover = crossovers[1]
    else:
        crossover = crossovers[0]
    return crossover


def detect_crossing_points(ab_rel_pow, g_rel_pow):
    crossing_points = []

    # Iterate through the series
    for i in range(1, len(ab_rel_pow)):
        # Check if the series cross each other
        if (ab_rel_pow[i] > g_rel_pow[i] and ab_rel_pow[i - 1] < g_rel_pow[i - 1]) or \
                (ab_rel_pow[i] < g_rel_pow[i] and ab_rel_pow[i - 1] > g_rel_pow[i - 1]):
            crossing_points.append(i)

    return crossing_points


def all_layers_ROI_map(layer_len, n_surf, ROI_indexes):
    return np.array([i[ROI_indexes] for i in np.split(np.arange(layer_len*n_surf), n_surf)]).flatten()


def fooofinator_par(freqs, psd, f_lims, n_jobs=-1):
    start_params=np.arange(f_lims[0],f_lims[1]-5,1)
    
    def run_fooof(i):
        start=start_params[i]
        fg=FOOOF(aperiodic_mode='fixed')
        fg.fit(freqs,psd, [start,f_lims[1]])
        if fg.has_model:
            ap_params=fg.get_params('aperiodic_params')
            return gen_aperiodic(freqs, ap_params)
        else:
            return np.zeros(psd.shape)*np.nan
    aperiodic = Parallel(
        n_jobs=n_jobs
    )(delayed(run_fooof)(i) for i in range(len(start_params)))
    return np.nanmedian(np.array(aperiodic),axis=0)


def fooofinator(freqs, psd, f_lims):
    start_params=np.arange(f_lims[0],f_lims[1]-5,1)
    test_aperiodic=[]
    for i,start in enumerate(start_params):
        fg=FOOOF(aperiodic_mode='fixed')
        fg.fit(freqs,psd, [start,f_lims[1]])
        if fg.has_model:
            ap_params=fg.get_params('aperiodic_params')
            test_aperiodic.append(gen_aperiodic(freqs, ap_params))
    return np.median(np.array(test_aperiodic),axis=0)