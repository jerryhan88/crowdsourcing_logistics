from init_project import *

O_GL = True
X_GL = False

def record_logs(fpath, contents):
    if fpath:
        with open(fpath, 'a') as f:
            f.write(contents)
    else:
        print(contents)