import itertools as it
import subprocess as sp
from joblib import Parallel, delayed


def job_to_do(subj_id, file_id):
    sp.run([
        "python /home/mszul/git/DANC_multilayer_laminar/01_source_power_per_vertex.py {} {}".format(subj_id, file_id)
    ], shell=True)

sub_f = list(it.product(range(0,9), [2,3]))

Parallel(n_jobs=len(sub_f))(delayed(job_to_do)(subj_id, file_id) for subj_id, file_id in sub_f)

