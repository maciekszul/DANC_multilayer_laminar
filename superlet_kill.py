import new_files
import subprocess as sp
from joblib import Parallel, delayed

dir_search = new_files.Files()

directory = "/home/mszul/datasets/explicit_implicit_beta/derivatives/processed"

SL = dir_search.get_files(
    directory, "*.npy", strings=["motor-epo", "visual-epo"], prefix="0", check="any"
)

lensl = len(SL)

def job_to_do(file_path):
    sp.run([
        "rm -f {}".format(file_path)
    ], shell=True)

Parallel(n_jobs=60)(delayed(job_to_do)(file_path) for file_path in SL)