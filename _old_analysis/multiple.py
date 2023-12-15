import sys
import subprocess as sp

try:
    sub_ix = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

script_path = "/home/mszul/git/DANC_multilayer_laminar/02_source_csd_per_vertex.py"


for i in range(24):
    try:
        sp.call(["python", script_path, str(sub_ix), str(i)])
    except IndexError:
        break