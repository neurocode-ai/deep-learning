# use this file at your own risk, if you have important work in your ipynb checkpoints
# then don't use it, or just remove that check in the conditional...
# i hate the checkpoints, so i always want to remove them, the bloat the repo
import sys
import os
import shutil

for dpath, dnames, fnames in os.walk(os.getcwd()):
    for dname in dnames:
        if dname in ['__pycache__', '.ipynb_checkpoints']:
            shutil.rmtree(os.path.join(dpath, dname))

