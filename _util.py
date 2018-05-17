import os.path as opath
import os
import datetime
import pickle
import csv


def log2file(fpath, contents):
    with open(fpath, 'a') as f:
        f.write(contents)


def write_log(fpath, contents):
    with open(fpath, 'a') as f:
        logContents = '\n\n'
        logContents += '======================================================================================\n'
        logContents += '%s\n' % str(datetime.datetime.now())
        logContents += '%s\n' % contents
        logContents += '======================================================================================\n'
        f.write(logContents)


def res2file(fpath, objV, gap, eliCpuTime, eliWallTime):
    with open(fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['objV', 'Gap', 'eliCpuTime', 'eliWallTime']
        writer.writerow(header)
        writer.writerow([objV, gap, eliCpuTime, eliWallTime])


def prb2file(fpath, _object):
    with open(fpath, 'wb') as fp:
        pickle.dump(_object, fp)


def bpt2file(fpath, contents=[]):
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

def itr2file(fpath, contents=[]):
    if not contents:
        if opath.exists(fpath):
            os.remove(fpath)
        with open(fpath, 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            header = ['itrNum', 'relObjV', 'selBC', 'selBC_RC', 'newBC', 'newBC_RC', 'eliCpuTime', 'eliWallTime']
            writer.writerow(header)
    else:
        with open(fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow(contents)

def set_grbSettings(m, grb_settings):
    for k, v in grb_settings.items():
        m.setParam(k, v)


def get_routeFromOri(edges, nodes):
    visited, adj = {}, {}
    for i in nodes:
        visited[i] = False
        adj[i] = []
    for i, j in edges:
        adj[i].append(j)
    route = []
    for n in nodes:
        if n.startswith('ori'):
            cNode = n
            break
    else:
        assert False
    while not cNode.startswith('dest'):
        visited[cNode] = True
        neighbors = [j for j in adj[cNode] if not visited[j]]
        route.append((cNode, neighbors[0]))
        cNode = neighbors[0]
        if visited[cNode]:
            break
    return route



