import sys
import json
import os.path as op
import nibabel as nb
import pandas as pd
import numpy as np
from os import sep
import new_files
from utilities import files
from scipy.spatial.distance import euclidean
from mne.time_frequency import psd_array_welch
from mne import read_epochs, pick_types
from tqdm import tqdm
from tools import *
import time


try:
    sub_ix = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    file_ix = int(sys.argv[2])
except:
    print("incorrect arguments")
    sys.exit()
    
try:
    json_file = sys.argv[3]
    print("USING:", json_file)
except:
    json_file = "settings.json"
    print("USING:", json_file)


start = time.monotonic()
# opening a json file
with open(json_file) as pipeline_file:
    info = json.load(pipeline_file)


dir_search = new_files.Files()


subject_paths = files.get_folders_files(info["dataset_dir"])[0]
subject_path = subject_paths[sub_ix]
subject = subject_path.split(sep)[-1]
subject_surf_path = op.join(info["anatomy_dir"], subject, "fs", "surf")
subject_proc_path = op.join(subject_path, info["which_dataset"])
subject_inv_path = op.join(subject_proc_path, "inverted")

## info file prep
info["fsnat_sphere_paths"] = dir_search.get_files(subject_surf_path, "*.gii", strings=["sphere", "reg"])
info["subject_path"] = subject_proc_path
info["pial"] = op.join(subject_surf_path, "pial.gii")
info["white"] = op.join(subject_surf_path, "white.gii")
info["pial_ds"] = op.join(subject_proc_path, "pial.ds.gii")
info["white_ds"] = op.join(subject_proc_path, "white.ds.gii")
info["pial_ds_nodeep"] = op.join(subject_proc_path, "pial.ds.link_vector.nodeep.gii")
info["white_ds_nodeep"] = op.join(subject_proc_path, "white.ds.link_vector.nodeep.gii")
info["pial_ds_nodeep_inflated"] = op.join(subject_proc_path, "pial.ds.inflated.nodeep.gii")
info["multilayer"] = files.get_files(subject_proc_path, info["which_dataset"], ".ds.link_vector.nodeep.gii")[2][0]
info["n_surf"] = int(info["which_dataset"].split("_")[-1])
info["atlas_labels_path"] = op.join(subject_proc_path, "atlas_glasser2016_labels.npy")
info["atlas_colors_path"] = op.join(subject_proc_path, "atlas_glasser2016_colors.npy")
info["cortical_thickness_path"] = op.join(subject_proc_path, "cortical_thickness.npy")
info["big_brain_layers_path"] = op.join(subject_proc_path, "big_brain_layers.json")
all_sensor_files = dir_search.get_files(info["dataset_dir"], "*.fif", strings=["autoreject", subject])
all_sensor_files_core_names = [i.split(sep)[-1].split(".")[0] for i in all_sensor_files]

all_MU_files = dir_search.get_files(
    subject_inv_path, "*.tsv", prefix="MU", 
    strings=[*all_sensor_files_core_names], check="any"
)
all_MU_files_core_names = [i.split(sep)[-1].split("_")[-1].split(".")[0] for i in all_MU_files]
mu_epo_map = np.isin(all_sensor_files_core_names, all_MU_files_core_names)
all_sensor_files = np.array(all_sensor_files)[mu_epo_map]
info["sensor_epochs_paths"] = all_sensor_files.tolist()
info["MU_paths"] = all_MU_files

## atlas + thicc map preps/loads
# atlas
if not all([op.exists(info["atlas_labels_path"]), op.exists(info["atlas_colors_path"])]):
    atlas_labels, atlas_colors = transform_atlas(
        info["annot_paths"], info["fsavg_sphere_paths"],
        info["fsnat_sphere_paths"], info["pial"], 
        info["pial_ds"], info["pial_ds_nodeep"]
    )
    np.save(info["atlas_labels_path"], atlas_labels)
    np.save(info["atlas_colors_path"], atlas_colors)
elif all([op.exists(info["atlas_labels_path"]), op.exists(info["atlas_colors_path"])]):
    atlas_labels = np.load(info["atlas_labels_path"])
    atlas_colors = np.load(info["atlas_colors_path"])

# cortical thickness from multilayer

multilayer = nb.load(info["multilayer"])
vertices, faces, normals = multilayer.agg_data()
vertices = np.split(vertices, info["n_surf"], axis=0)
faces = np.split(faces, info["n_surf"], axis=0)
normals = np.split(normals, info["n_surf"], axis=0)

if not op.exists(info["cortical_thickness_path"]):
    cortical_thickness = np.array([euclidean(vertices[0][vx], vertices[-1][vx]) for vx in range(vertices[0].shape[0])])
    np.save(info["cortical_thickness_path"], cortical_thickness)
elif op.exists(info["cortical_thickness_path"]):
    cortical_thickness = np.load(info["cortical_thickness_path"])

# Big Brain layer boundaries
if not op.exists(info["big_brain_layers_path"]):
    bb_l_paths = dir_search.get_files(info["big_brain_path"], "*.gii", strings=["hemi-L"], prefix="tpl-fsaverage")
    bb_r_paths = dir_search.get_files(info["big_brain_path"], "*.gii", strings=["hemi-R"], prefix="tpl-fsaverage")
    bb_lr_paths = list(zip(bb_l_paths, bb_r_paths))
    fsavg_values = {i+1: [nb.load(i).agg_data() for i in bb_lr_paths[i]] for i in range(len(bb_lr_paths))}
    fsnat_ds_values = {i: fsavg_vals_to_native(
        fsavg_values[i],
        info["fsavg_sphere_paths"],
        info["fsnat_sphere_paths"], 
        info["pial"], 
        info["pial_ds"], 
        info["pial_ds_nodeep"]
    ) for i in fsavg_values.keys()}
    overall_thickness = np.sum(np.vstack([fsnat_ds_values[i] for i in fsnat_ds_values.keys()]), axis=0)
    bb_bound_prop = {i: np.divide(
        layers_fsnative_ds_values[i], 
        overall_thickness, 
        where=overall_thickness != 0
    ) for i in layers_fsnat_ds_values.keys()}
    bb_bound_prop = {i: list(bb_bound_prop[i].astype(float)) for i in bb_bound_prop.keys()}
    with open(info["big_brain_layers_path"], "w") as fp:
        json.dump(bb_bound_prop, fp, indent=4)
if op.exists(info["big_brain_layers_path"]):
    with open(info["big_brain_layers_path"], "r") as fp:
        bb_bound_prop = json.load(fp)
    bb_bound_prop = {i: np.array(bb_bound_prop[i]) for i in bb_bound_prop.keys()}

# computation
fif_MU = list(zip(info["sensor_epochs_paths"], info["MU_paths"]))
fif, MU = fif_MU[file_ix]
core_name = all_MU_files_core_names[file_ix]
epo_type = [i for i in ["motor", "visual"] if i in core_name][0]

fif = read_epochs(fif, verbose=False)
fif = fif.pick_types(meg=True, ref_meg=False, misc=False)
sfreq = fif.info["sfreq"]
fif = fif.get_data()
new_mu_path = MU.split(".")[0] + ".npy"
if not op.exists(new_mu_path):
    MU = pd.read_csv(MU, sep="\t", header=None).to_numpy()
    np.save(new_mu_path, MU)
elif op.exists(new_mu_path):
    MU = np.load(new_mu_path)

flims = [0.1,125]

psd_mean = []

with tqdm(total=MU.shape[0], position=0, leave=True) as pbar:
    for vertex in tqdm(range(MU.shape[0]), position=0, leave=True):
        vertex_source = []
        for trial in fif:
             vertex_source.append(np.dot(trial.T, MU[vertex]))
        vertex_source = np.array(vertex_source)
        winsize = int(sfreq)
        overlap = int(winsize/2)
        psd, freqs = psd_array_welch(
            vertex_source, sfreq, fmin=flims[0], 
            fmax=flims[1], n_fft=2000, 
            n_overlap=overlap, n_per_seg=winsize,
            window="hann", verbose=False, n_jobs=1
        )
        mean_psd = np.nanmean(psd,axis=0)
        psd_mean.append(mean_psd)
        pbar.update()


psd_path = op.join(subject_inv_path, "PSD_" + core_name + ".npy")
np.save(psd_path, np.array(psd_mean))

info_path = op.join(subject_proc_path, "info.json")
with open(info_path, "w") as fp:
    json.dump(info, fp, indent=4)

stop = time.monotonic()
duration = int((stop - start)/60.0)
print("{} | {} | finished processing in ~{} minutes.".format(subject, core_name, duration))