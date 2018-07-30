import os.path as opath
import multiprocessing
import os
import csv, pickle
from geopy.distance import vincenty
from shapely.geometry import Polygon, Point
from random import seed, choice, randrange
import pandas as pd
import numpy as np
from functools import reduce
#
from __path_organizer import exp_dpath
from mrtScenario import PER25, PER75, STATIONS

MIN60, SEC60 = 60.0, 60.0
MIN20 = 20
Meter1000 = 1000.0
WALKING_SPEED = 5.0  # km/hour


def gen_instances4TT(problem_dpath):
    from mrtScenario import inputConvertPickle
    if not opath.exists(problem_dpath):
        os.mkdir(problem_dpath)
        for dname in ['dplym', 'prmt']:
            os.mkdir(opath.join(problem_dpath, dname))
    #
    min_durPD = 20
    minTB, maxTB = 2, 4
    flowPER, detourPER = PER75, PER25
    #
    #  '4small', '5out', '7inter', '11interOut'
    stationSel = '11interOut'
    stations = STATIONS[stationSel]


    # 75, , 105, 120

    numTasks = 300
    thDetour = 90

    for d2d_ratio in np.arange(0.0, 1.1, 0.25):
        for seedNum in range(10):
            problemName = '%s-nt%d-d2dR%d-dt%d-sn%d' % (stationSel, numTasks, d2d_ratio * 100,
                                                        thDetour, seedNum)
            seed(seedNum)
            np.random.seed(seedNum)
            #
            flow_oridest = [(stations[i], stations[j])
                            for i in range(len(stations)) for j in range(len(stations)) if i != j]
            numBundles = int(numTasks / ((minTB + maxTB) / 2)) + 1
            task_ppdp, tasks, \
            flows, \
            travel_time, \
            _, minWS = gen_instance(flow_oridest, numTasks, d2d_ratio, min_durPD, detourPER, flowPER)
            problem = [problemName,
                       flows, tasks,
                       numBundles, minTB, maxTB,
                       travel_time, thDetour,
                       minWS]
            inputConvertPickle(problem, flow_oridest, task_ppdp, problem_dpath)


def gen_instance(flow_oridest, numTasks, d2d_ratio, min_durPD, detourPER, flowPER):
    from sgMRT import get_MRT_district, get_mrtNetNX, get_route, get_coordMRT
    from sgDistrict import get_districtPopPoly
    from mrtScenario import get_travelTimeSG, handle_locNtt_MRT, gen_taskPD_onRoute
    from mrtScenario import aDay_EZ_fpath
    #
    d2d_nt = int(numTasks * d2d_ratio)
    b2b_nt = numTasks - d2d_nt
    #
    distPop, distPoly = get_districtPopPoly()
    distPoly_sh = {dn: Polygon(poly) for dn, poly in distPoly.items()}
    #
    MRT_district = get_MRT_district()
    mrt_coords = get_coordMRT()
    mrtNetNX = get_mrtNetNX()
    flow_route, flow_distPop = {}, {}
    for ori, dest in flow_oridest:
        route = get_route(mrtNetNX, ori, dest)
        flow_route[ori, dest] = route
        flow_distPop[ori, dest] = [(MRT_district[MRT], distPop[MRT_district[MRT]]) for MRT in route]
    #
    task_ppdp_d2d = []
    MRTs_tt, locPD_MRT_tt = get_travelTimeSG()
    while len(task_ppdp_d2d) < d2d_nt:
        i = randrange(len(flow_oridest))
        ori, dest = flow_oridest[i]
        distNames, distPops = map(np.array, zip(*flow_distPop[ori, dest]))
        distWeights = distPops / distPops.sum()
        dn = np.random.choice(distNames, p=distWeights)
        j = distNames.tolist().index(dn)
        if j == len(flow_distPop) - 1:
            continue
        lats, lngs = map(np.array, zip(*distPoly[dn]))
        lat0 = np.random.uniform(lats.min(), lats.max())
        lng0 = np.random.uniform(lngs.min(), lngs.max())
        if not Point(lat0, lng0).within(distPoly_sh[dn]):
            continue
        #
        distNames1, distPops1 = map(np.array, zip(*flow_distPop[ori, dest][j:]))
        distWeights1 = distPops1 / distPops1.sum()
        dn = np.random.choice(distNames1, p=distWeights1)
        k = distNames.tolist().index(dn)
        lats, lngs = map(np.array, zip(*distPoly[dn]))
        lat1 = np.random.uniform(lats.min(), lats.max())
        lng1 = np.random.uniform(lngs.min(), lngs.max())
        if not Point(lat1, lng1).within(distPoly_sh[dn]):
            continue
        assert j <= k
        if j == k:
            continue
        distance = vincenty((lat0, lng0), (lat1, lng1)).km
        duration = (distance / WALKING_SPEED) * MIN60
        if duration < min_durPD:
            continue
        nMRT0, nMRT1 = flow_route[ori, dest][j], flow_route[ori, dest][k]
        loc0, loc1 = '%f_%f' % (lat0, lng0), '%f_%f' % (lat1, lng1)
        dur0 = (vincenty((lat0, lng0), tuple(mrt_coords[nMRT0])).km / WALKING_SPEED) * MIN60
        dur1 = (vincenty((lat1, lng1), tuple(mrt_coords[nMRT1])).km / WALKING_SPEED) * MIN60
        locPD_MRT_tt[loc0] = (dur0, nMRT0)
        locPD_MRT_tt[loc1] = (dur1, nMRT1)
        task_ppdp_d2d.append((loc0, loc1))
    #
    task_ppdp_b2b = gen_taskPD_onRoute(flow_oridest, b2b_nt, min_durPD)
    task_ppdp = task_ppdp_d2d + task_ppdp_b2b
    #
    (numLocs, lid_loc, loc_lid), travel_time = handle_locNtt_MRT(flow_oridest, task_ppdp, locPD_MRT_tt)
    detourTimes = []
    for ori, dest in flow_oridest:
        iori, idest = [loc_lid.get(k) for k in [ori, dest]]
        for pp, dp in task_ppdp:
            ipp, idp = [loc_lid.get(k) for k in [pp, dp]]
            detourTimes.append(travel_time[iori][ipp] + travel_time[ipp][idp] + travel_time[idp][idest] \
                               - travel_time[iori][idest])
    thDetour = np.percentile(detourTimes, detourPER)
    #
    flow_count = {}
    with open(aDay_EZ_fpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            fSTN, tSTN = [row[cn] for cn in ['fSTN', 'tSTN']]
            count = eval(row['count'])
            if (fSTN, tSTN) in flow_oridest:
                flow_count[loc_lid[fSTN], loc_lid[tSTN]] = count
    flows = []
    for fSTN, tSTN in flow_oridest:
        flows.append([(loc_lid[fSTN], loc_lid[tSTN]), flow_count[loc_lid[fSTN], loc_lid[tSTN]]])
    assert len(flows) == len(flow_oridest)
    counts = np.array([count for _, count in flows])
    weights = counts / counts.sum()
    minWS = np.percentile(weights, flowPER)
    tasks = []
    for i, (loc0, loc1) in enumerate(task_ppdp):
        tasks.append((i, loc_lid[loc0], loc_lid[loc1]))
    #
    return task_ppdp, tasks, flows, travel_time, thDetour, minWS


def summaryRD():
    def process_files(fns, wid, wsDict):
        rows = []
        for fn in fns:
            _, prefix = fn[:-len('.pkl')].split('_')
            prmt_fpath = opath.join(prmt_dpath, fn)
            with open(prmt_fpath, 'rb') as fp:
                prmt = pickle.load(fp)
            d2d_ratio = float(fn[:-len('.pkl')].split('-')[-3][len('d2dR'):]) / 100
            K, T, _delta, cW = [prmt.get(k) for k in ['K', 'T', '_delta', 'cW']]
            new_row = [prefix, len(K), len(T), d2d_ratio, _delta, cW]
            aprc = 'CWL4'
            sol_fpath = opath.join(sol_dpath, 'sol_%s_%s.csv' % (prefix, aprc))
            log_fpath = opath.join(log_dpath, '%s_itr%s.csv' % (prefix, aprc))
            if opath.exists(sol_fpath):
                with open(sol_fpath) as r_csvfile:
                    reader = csv.DictReader(r_csvfile)
                    for row in reader:
                        objV, eliCpuTime = [row[cn] for cn in ['objV', 'eliCpuTime']]
                    new_row += [objV, eliCpuTime]
            else:
                new_row += ['-', '-']
            rows.append(new_row)
        wsDict[wid] = rows
    #
    summaryPC_dpath = opath.join(exp_dpath, '_TaskType')
    rd_fpath = reduce(opath.join, [summaryPC_dpath, 'rawDataTT.csv'])
    with open(rd_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['pn', 'numPaths', 'numTasks', 'd2d_ratio', 'thDetour', 'thWS',
                  'objV', 'cpuT']
        writer.writerow(header)
    #
    prmt_dpath = reduce(opath.join, [summaryPC_dpath, 'prmt'])
    sol_dpath = reduce(opath.join, [summaryPC_dpath, 'sol'])
    log_dpath = reduce(opath.join, [summaryPC_dpath, 'log'])

    numProcessors = multiprocessing.cpu_count()
    worker_fns = [[] for _ in range(numProcessors)]
    prmt_fns = sorted([fn for fn in os.listdir(prmt_dpath) if fn.endswith('.pkl')])
    for i, fn in enumerate(prmt_fns):
        worker_fns[i % numProcessors].append(fn)
    ps = []
    wsDict = multiprocessing.Manager().dict()
    for wid, fns in enumerate(worker_fns):
        p = multiprocessing.Process(target=process_files,
                                    args=(fns, wid, wsDict))
        ps.append(p)
        p.start()
    for p in ps:
        p.join()
    for _, rows in wsDict.items():
        for row in rows:
            with open(rd_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow(row)
    #
    df = pd.read_csv(rd_fpath)
    df['seedNum'] = df.apply(lambda row: int(row['pn'].split('-')[-1][len('sn'):]), axis=1)
    df = df.sort_values(by=['d2d_ratio', 'seedNum'])
    df = df.drop(['seedNum'], axis=1)
    df.to_csv(rd_fpath, index=False)


def summaryTT():
    summaryPP_dpath = opath.join(exp_dpath, '_TaskType')
    rd_fpath = reduce(opath.join, [summaryPP_dpath, 'rawDataTT.csv'])
    sum_fpath = reduce(opath.join, [summaryPP_dpath, 'summaryTT.csv'])
    odf = pd.read_csv(rd_fpath)
    odf = odf.drop(['pn', 'thWS'], axis=1)
    odf = odf.replace('-', np.nan)
    odf[odf.columns] = odf[odf.columns].apply(pd.to_numeric)
    df = odf.groupby(['d2d_ratio', 'thDetour']).mean().reset_index()
    # aprcs = ['CWL%d' % cwl_no for cwl_no in range(1, 6)] + ['GH']
    # sdf = odf.groupby(['numPaths', 'numTasks']).std().reset_index()
    # for aprc in aprcs:
    #     df['%s_cpuT_sd' % aprc] = sdf['%s_cpuT' % aprc]

    df.to_csv(sum_fpath, index=False)




if __name__ == '__main__':
    # gen_instances4TT(opath.join(exp_dpath, 'm45000'))
    gen_instances4TT(opath.join(exp_dpath, 'm900'))
    # summaryRD()
    # summaryTT()