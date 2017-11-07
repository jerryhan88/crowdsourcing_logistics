from init_project import *

O_GL, X_GL = True, False
LOGGING_FEASIBILITY = False


def record_logs(fpath, contents):
    if fpath:
        with open(fpath, 'a') as f:
            f.write(contents)
    else:
        print(contents)