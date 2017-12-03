import os.path as opath
import os
import pickle
import csv

O_GL, X_GL = True, False
LOGGING_FEASIBILITY = False


def record_log(fpath, contents):
    if fpath:
        with open(fpath, 'a') as f:
            f.write(contents)
    else:
        print(contents)


def record_problem(fpath, _object):
    with open(fpath, 'wb') as fp:
        pickle.dump(_object, fp)


def record_bpt(fpath, contents=[]):
    if not contents:
        if opath.exists(fpath):
            os.remove(fpath)
        with open(fpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            header = ['nid', 'startWallTime', 'eliWallTime', 'eliCpuTime', 'eventType', 'contents']
            writer.writerow(header)
    else:
        with open(fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow(contents)


def record_res(fpath, objV, gap, eliCpuTime, eliWallTiem):
    with open(fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['objV', 'Gap', 'eliCpuTime', 'eliWallTime']
        writer.writerow(header)
        writer.writerow([objV, gap, eliCpuTime, eliWallTiem])