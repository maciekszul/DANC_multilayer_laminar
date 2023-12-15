import sys
from pathlib import Path
from utilities import files
import os.path as op
import numpy as np
from os import sep, remove
import itertools as it
import json
from mne import read_epochs
import matlab.engine
import shutil
import time
import new_files


try:
    sub_ix = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    json_file = sys.argv[3]
    print("USING:", json_file)
except:
    json_file = "settings.json"
    print("USING:", json_file)


def many_is_in(multiple, target):
    check_ = []
    for i in multiple:
        check_.append(i in target)
    return any(check_)

def all_is_in(multiple, target):
    check_ = []
    for i in multiple:
        check_.append(i in target)
    return all(check_)

def cat(options, target):
    for i in options:
        if i in target:
            return i

def get_res4(ds_path, sub, ses, run):
    ref_path = op.join(ds_path, "raw", sub, ses, "meg")
    blocks = files.get_folders_files(ref_path)[0]
    block = [i for i in blocks if "block-{}".format(run[1:]) in i][0]
    res4_path = files.get_files(block, "", ".res4")[2][0]
    return res4_path


def average_filter_convert(file_path, ds_path, parasite, filt=False, l_freq=None, h_freq=None):
    path_split = file_path.split(sep)
    filename_core = path_split[-1].split(".")[0]
    sub = filename_core[11:18]
    ses = filename_core[19:25]
    run = filename_core[26:29]
    res4_path = get_res4(ds_path, sub, ses, run)
    dir_path = str(sep).join(path_split[:-1] + ["avg_spm", ""])
    files.make_folder(dir_path)
    
    filt_status = "_no_filter"
    if filt:
        filt_status = "_filt"
    
    output_file = "spm-converted{}_{}".format(filt_status, filename_core)
    output_path = op.join(dir_path, output_file)
    average_file = output_path + "-ave.fif"
    mat_output = output_path + ".mat"
    if not op.exists(mat_output):
        if not op.exists(average_file):
            epochs = read_epochs(file_path, verbose=False)
            epochs = epochs.average()
            if filt:
                epochs.filter(l_freq=l_freq, h_freq=h_freq)
            epochs.save(average_file)

        parasite.convert_mne_to_spm(res4_path, average_file, mat_output, 0, nargout=0)
        if op.isfile(average_file):
            remove(average_file)
        else:
            print(average_file, "does not exists")
        print(filename_core, "converted")

    else:
        print(mat_output, "exists")
        
    return mat_output


def invert_multisurface(inverted_output, t1_file, mat_file, subjects_info, parasite, layers=11):
    """
    inverted_output 
    """
    files.make_folder(inverted_output)
    input_path = Path(mat_file)
    bits = input_path.name.split("_")[-1].split("-")
    sub = "-".join([bits[1], bits[2]])
    epo = bits[6]
    ses = "-".join([bits[3], bits[4]])
    run = bits[5]
    link_vector = files.get_files(
        op.join(Path(inverted_output).parent), 
        "multilayer", ".ds.link_vector.nodeep.gii"
    )[2][0]
    mu_file = op.join(inverted_output, "MU_" + input_path.stem + ".tsv")
    it_file = op.join(inverted_output, "IT_" + input_path.stem + ".tsv")
    res_file = op.join(inverted_output, "res_" + input_path.stem + ".tsv")
    json_out_file = op.join(inverted_output, "invert-res_" + input_path.stem + ".json")
    parasite.invert_multisurface(
        inverted_output, subjects_info, mat_file, t1_file, 
        link_vector, mu_file, it_file, res_file, 
        json_out_file, str(input_path.stem), float(layers), sub, ses, run, epo
    )


start = time.monotonic()
# opening a json file
with open(json_file) as pipeline_file:
    info = json.load(pipeline_file)


# setup paths
dir_search = new_files.Files()

subject_paths = files.get_folders_files(info["dataset_dir"])[0]
subject_path = subject_paths[sub_ix]
subject = subject_path.split(sep)[-1]

layers_name = "multilayer_{}".format(str(info["layers"]).zfill(2))

output_folder = op.join(subject_path, layers_name)
files.make_folder(output_folder)

fs_path = op.join(info["anatomy_dir"], subject, "fs")

parasite = matlab.engine.start_matlab()
# create multilayer
parasite.create_multilayer_surface(
    fs_path, info["layers"], output_folder, 
    layers_name, info["freesurfer_singularity"], "/home/common/bonaiuto/",
    nargout=0
)

# prepare surfaces
parasite.prepare_surfaces(
    fs_path, output_folder,
    nargout=0
)

# prepare multilayer
parasite.prepare_multilayer_surface(
    fs_path, output_folder, float(info["layers"]), layers_name,
    nargout=0
)

# intermediate steps moving
intm_path = op.join(output_folder, "inter")
files.make_folder(intm_path)
files_paths = files.get_files(output_folder, "", "")[2]
lyrs = np.arange(info["layers"])[1:-1]
lyrs = [str(i)for i in lyrs]
files_input = [i for i in files_paths if many_is_in(lyrs, i.split(sep)[-1])]
files_input = [i for i in files_input if layers_name not in i.split(sep)[-1]]
files_output = [op.join(intm_path, i.split(sep)[-1]) for i in files_paths if many_is_in(lyrs, i.split(sep)[-1])]
files_io = list(zip(files_input, files_output))
[shutil.move(*i) for i in files_io];

# get the MEG files

exclude=[
    "sub-002-ses-01",
    "sub-002-ses-02",
    "sub-006-ses-01",
    "sub-006-ses-02",
    "sub-008-ses-01"
]

session_files = dir_search.get_files(subject_path, "*.fif", prefix="autoreject")
session_files = [i for i in session_files if not many_is_in(exclude, i)]

mat_files = [average_filter_convert(i, info["old_dataset"], parasite) for i in session_files]

inverted_output = op.join(output_folder, "inverted")

t1_file = info["T1_dir"].format(subject)

[invert_multisurface(inverted_output, t1_file, i, info["subject_info"], parasite, layers=info["layers"]) for i in mat_files]

stop = time.monotonic()
duration = np.round((stop - start)/60.0, 2)
print("{} | finished processing in ~{} minutes.".format(subject, duration))