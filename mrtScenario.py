import os.path as opath
import csv, pickle
import numpy as np
from random import seed, choice
#
from __path_organizer import ez_dpath, pf_dpath

aDayNight_EZ_fpath = opath.join(ez_dpath, 'EZ-MRT-D20130801-H18H23.csv')
aDay_EZ_fpath = opath.join(ez_dpath, 'EZ-MRT-D20130801.csv')
seed(0)

PER25, PER50, PER75 = np.arange(25, 100, 25)
STATIONS = {
    '4small': ['Paya Lebar', 'Raffles Place', 'Bishan', 'Dhoby Ghaut'],
    '5out': ['Raffles Place', 'Tampines', 'Yishun', 'Jurong East', 'Choa Chu Kang'],
    '7inter': ['Paya Lebar', 'Raffles Place', 'HarbourFront', 'Buona Vista', 'Bishan', 'Serangoon', 'Dhoby Ghaut'],
    '11interOut': ['Tampines', 'Yishun', 'Jurong East', 'Choa Chu Kang',
               'Paya Lebar', 'Raffles Place', 'HarbourFront', 'Buona Vista', 'Bishan', 'Serangoon', 'Dhoby Ghaut'],
}


def convert_prob2prmt(problemName,
                      flows, tasks,
                      numBundles, minTB, maxTB,
                      numLocs, travel_time, thDetour,
                      minWS):
    B = list(range(numBundles))
    cB_M, cB_P = minTB, maxTB
    cW = minWS
    T = list(range(len(tasks)))
    iPs, iMs = list(zip(*[(tasks[i][1], tasks[i][2]) for i in T]))
    P, D = set(), set()
    _N = {}
    for i in T:
        P.add('p%d' % i)
        D.add('d%d' % i)
        #
        _N['p%d' % i] = iPs[i]
        _N['d%d' % i] = iMs[i]
    #
    # Path
    #
    c_k, temp_OD = [], []
    if type(flows) == list:
        for i in range(numLocs):
            for j in range(numLocs):
                if flows[i][j] == 0:
                    continue
                c = flows[i][j]
                temp_OD.append((i, j))
                c_k.append(c)
    else:
        assert type(flows) == dict
        for (i, j), c in flows.items():
            if c == 0:
                continue
            temp_OD.append((i, j))
            c_k.append(c)
    sumC = sum(c_k)
    K = list(range(len(c_k)))
    w_k = [c_k[k] / float(sumC) for k in K]
    _kP, _kM = list(zip(*[temp_OD[k] for k in K]))
    _delta = thDetour
    t_ij = {}
    for k in K:
        kP, kM = 'ori%d' % k, 'dest%d' % k
        t_ij[kP, kP] = travel_time[_kP[k]][_kP[k]]
        t_ij[kM, kM] = travel_time[_kM[k]][_kM[k]]
        t_ij[kP, kM] = travel_time[_kP[k]][_kM[k]]
        t_ij[kM, kP] = travel_time[_kM[k]][_kP[k]]
        for i in _N:
            t_ij[kP, i] = travel_time[_kP[k]][_N[i]]
            t_ij[i, kP] = travel_time[_N[i]][_kP[k]]
            #
            t_ij[kM, i] = travel_time[_kM[k]][_N[i]]
            t_ij[i, kM] = travel_time[_N[i]][_kM[k]]
    for i in _N:
        for j in _N:
            t_ij[i, j] = travel_time[_N[i]][_N[j]]
    N = set(_N.keys())
    #
    return {'problemName': problemName,
            'bB': numBundles,
            'B': B, 'cB_M': cB_M, 'cB_P': cB_P,
            'K': K, 'w_k': w_k,
            'T': T, 'P': P, 'D': D, 'N': N,
            't_ij': t_ij, '_delta': _delta,
            'cW': cW}



def gen_instance(stations, numTasks, min_durPD, detourPER, flowPER):
    flow_oridest = [(stations[i], stations[j])
                    for i in range(len(stations)) for j in range(len(stations)) if i != j]
    locPD_durMRT = []
    with open(opath.join(pf_dpath, 'tt-MRT-LocationPD.csv')) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            nearestMRT, Location = [row[cn] for cn in ['nearestMRT', 'Location']]
            Duration = eval(row['Duration'])
            if nearestMRT in stations:
                locPD_durMRT.append([Location, Duration, nearestMRT])
    MRTs_tt, locPD_MRT_tt = get_travelTimeSG()
    task_ppdp = []
    while len(task_ppdp) < numTasks:
        loc0, dur0, nMRT0 = choice(locPD_durMRT)
        loc1, dur1, nMRT1 = choice(locPD_durMRT)
        k = (nMRT0, nMRT1) if nMRT0 < nMRT1 else(nMRT1, nMRT0)
        if loc0 != loc1 and min_durPD < dur0 + MRTs_tt[k] + dur1:
            task_ppdp.append((loc0, loc1))
    #
    (numLocs, lid_loc, loc_lid), travel_time = handle_locNtt_MRT(flow_oridest, task_ppdp)
    detourTimes = []
    for ori, dest in flow_oridest:
        iori, idest = [loc_lid.get(k) for k in [ori, dest]]
        for pp, dp in task_ppdp:
            ipp, idp = [loc_lid.get(k) for k in [pp, dp]]
            detourTimes.append(travel_time[iori][ipp] + travel_time[ipp][idp] + travel_time[idp][idest] \
                               - travel_time[iori][idest])
    thDetour = np.percentile(detourTimes, detourPER)
    #
    flows = {}
    with open(aDay_EZ_fpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            fSTN, tSTN = [row[cn] for cn in ['fSTN', 'tSTN']]
            count = eval(row['count'])
            if (fSTN, tSTN) in flow_oridest:
                flows[loc_lid[fSTN], loc_lid[tSTN]] = count
    assert len(flows) == len(flow_oridest)
    counts = np.array(list(flows.values()))
    weights = counts / counts.sum()
    minWS = np.percentile(weights, flowPER)
    tasks = []
    for i, (loc0, loc1) in enumerate(task_ppdp):
        tasks.append((i, loc_lid[loc0], loc_lid[loc1]))
    #
    return flow_oridest, task_ppdp, flows, tasks, numLocs, travel_time, thDetour, minWS


def inputConvertPickle(problem, flow_oridest, task_ppdp, pkl_dir):
    problemName = problem[0]
    prmt = convert_prob2prmt(*problem)
    with open(opath.join(pkl_dir, 'prmts_%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(prmt, fp)
    vizInputs = [flow_oridest, task_ppdp]
    with open(opath.join(pkl_dir, 'dplym_%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(vizInputs, fp)
    #
    return prmt


def get_travelTimeSG():
    pkl_fpath = opath.join(pf_dpath, 'travelTimeSG.pkl')
    if not opath.exists(pkl_fpath):
        MRTs_tt, MRTs = {}, set()
        with open(opath.join(pf_dpath, 'travelTimeMRT.csv')) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                fSTN, tSTN = [row[cn] for cn in ['fSTN', 'tSTN']]
                travelTime = eval(row['travelTime'])
                MRTs_tt[fSTN, tSTN] = travelTime
                MRTs.add(fSTN)
                MRTs.add(tSTN)
        for mrt in MRTs:
            MRTs_tt[mrt, mrt] = 0.0
        locPD_MRT_tt = {}
        with open(opath.join(pf_dpath, 'tt-MRT-LocationPD.csv')) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                nearestMRT, Location = [row[cn] for cn in ['nearestMRT', 'Location']]
                Duration = eval(row['Duration'])
                locPD_MRT_tt[Location] = [Duration, nearestMRT]            
        with open(pkl_fpath, 'wb') as fp:
            pickle.dump([MRTs_tt, locPD_MRT_tt], fp)
    else:
        with open(pkl_fpath, 'rb') as fp:
            MRTs_tt, locPD_MRT_tt = pickle.load(fp)
    #
    return MRTs_tt, locPD_MRT_tt
            

def handle_locNtt_MRT(flow_oridest, task_ppdp):
    MRTs_tt, locPD_MRT_tt = get_travelTimeSG()
    lid_loc, loc_lid = {}, {}
    numLocs = 0
    for locs in flow_oridest + task_ppdp:
        for loc in locs:
            if loc not in loc_lid:
                lid_loc[numLocs] = loc
                loc_lid[loc] = numLocs
                numLocs += 1
    travel_time = [[0.0 for _ in range(numLocs)] for _ in range(numLocs)]
    for ori0, dest0 in flow_oridest:
        k0 = (ori0, dest0) if ori0 < dest0 else (dest0, ori0)
        travel_time[loc_lid[ori0]][loc_lid[dest0]] = MRTs_tt[k0]
        travel_time[loc_lid[dest0]][loc_lid[ori0]] = MRTs_tt[k0]
        for ori1, dest1 in flow_oridest:
            for loc0, loc1 in [(ori0, ori1), (ori0, dest1),
                               (dest0, ori1), (dest0, dest1)]:
                k1 = (loc0, loc1) if loc0 < loc1 else (loc1, loc0)
                tt = MRTs_tt[k1]
                travel_time[loc_lid[loc0]][loc_lid[loc1]] = tt
                travel_time[loc_lid[loc1]][loc_lid[loc0]] = tt
        for pp, dp in task_ppdp:
            pp_tt, pp_nMRT = locPD_MRT_tt[pp]
            dp_tt, dp_nMRT = locPD_MRT_tt[dp]
            for loc0, loc1, nMRT, w_tt in [(ori0, pp, pp_nMRT, pp_tt),
                                           (ori0, dp, dp_nMRT, dp_tt),
                                           (dest0, pp, pp_nMRT, pp_tt),
                                           (dest0, dp, dp_nMRT, dp_tt)]:
                k1 = (loc0, nMRT) if loc0 < nMRT else (nMRT, loc0)
                m_tt = MRTs_tt[k1]
                travel_time[loc_lid[loc0]][loc_lid[loc1]] = m_tt + w_tt
                travel_time[loc_lid[loc1]][loc_lid[loc0]] = m_tt + w_tt

    for pp0, dp0 in task_ppdp:
        pp0_tt, pp0_nMRT = locPD_MRT_tt[pp0]
        dp0_tt, dp0_nMRT = locPD_MRT_tt[dp0]
        k0 = (pp0_nMRT, dp0_nMRT) if pp0_nMRT < dp0_nMRT else (dp0_nMRT, pp0_nMRT)
        travel_time[loc_lid[pp0]][loc_lid[dp0]] = MRTs_tt[k0] + \
                                                  (pp0_tt + dp0_tt if pp0 != dp0 else 0.0)
        travel_time[loc_lid[dp0]][loc_lid[pp0]] = MRTs_tt[k0] + \
                                                  (pp0_tt + dp0_tt if pp0 != dp0 else 0.0)
        for pp1, dp1 in task_ppdp:
            for loc0, loc1 in [(pp0, pp1), (pp0, dp1),
                               (dp0, pp1), (dp0, dp1)]:
                loc0_tt, loc0_nMRT = locPD_MRT_tt[loc0]
                loc1_tt, loc1_nMRT = locPD_MRT_tt[loc1]
                k0 = (loc0_nMRT, loc1_nMRT) if loc0_nMRT < loc1_nMRT else (loc1_nMRT, loc0_nMRT)
                travel_time[loc_lid[loc0]][loc_lid[loc1]] = MRTs_tt[k0] + \
                                                            (loc0_tt + loc1_tt if loc0 != loc1 else 0.0)
                travel_time[loc_lid[loc1]][loc_lid[loc0]] = MRTs_tt[k0] + \
                                                            (loc0_tt + loc1_tt if loc0 != loc1 else 0.0)
    #
    return (numLocs, lid_loc, loc_lid), travel_time


def mrtS1(pkl_dir='_temp'):
    thDetour = 80
    problemName = 'mrtS1_dt%d' % thDetour
    numBundles, minTB, maxTB = 4, 2, 3
    minWS = 0.1
    
    flow_oridest = [  ('Raffles Place', 'Tampines'),
                    ('Raffles Place', 'Bedok'),
                    ('Tanjong Pagar', 'Tampines'),
                    #
                    ('Raffles Place', 'Bishan'),
                    ('Orchard', 'Yishun'),
                    ('Raffles Place', 'Ang Mo Kio'),
                    #
                    ('Jurong East', 'Choa Chu Kang'),
                    ('Jurong East', 'Boon Lay'),
                    ('Jurong East', 'Yew Tee'),
                    #
                    ('Tanjong Pagar', 'Lakeside'),
                    ('Raffles Place', 'Lakeside'),
                    ('Tanjong Pagar', 'Boon Lay'),  ]
    task_ppdp = [
        ('Hi-Tech Phone Centre Pte Ltd at 810 Geylang Road, #01-06', 'POPStation@Bedok Point'),
                # Paya Lebar -> Bedok
        ('Alfa Marketing at 1G Cantonment Road, #01-07, Pinnacle @ Duxton', 'POPStation@General Post Office'),
                # Tanjong Pagar -> Paya Lebar
        #
        ('Guardian at Raffles City Shopping Centre, #B1-01', 'POPStation@Junction 8'),
                # City Hall -> Bishan
        ('Bangkit Kiosk at 150 Orchard Road, #01-59', 'POPStation@Bishan CC'),
                # Somerset -> Bishan
        #
        ('Solular Plus (Jurong) at 131 Jurong Gateway Road, #01-255', 'POPStation@BukitBatokCentralPO'),
                # Jurong East -> Bukit Batok
        ('Grokars at Blk 225A, Jurong East Street 21, #01-K1', 'POPStation@BukitBatokCentralPO'),
                # Chinese Garden -> Bukit Batok
        #
        ('Guardian at Clifford Centre, #01-19/20/21', 'Ninja Box at Westgate'),
                # Raffles Place -> Jurong East
        ('Alfa Marketing at 1G Cantonment Road, #01-07, Pinnacle @ Duxton', 'Ninja Box at Star Vista'),
                # Tanjong Pagar -> Buona Vista
        #
        ('Guardian at Raffles City Shopping Centre, #B1-01', 'POPStation@Tanjong Pagar PO'),
                # Raffles Place -> Tanjong Pagar
        ('Alfa Marketing at 1G Cantonment Road, #01-07, Pinnacle @ Duxton', 'POPStation@Chinatown Point'),
                # Tanjong Pagar -> Chinatown
    ]
    #
    (numLocs, lid_loc, loc_lid), travel_time = handle_locNtt_MRT(flow_oridest, task_ppdp)
    #
    flows = {}
    with open(aDayNight_EZ_fpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            fSTN, tSTN = [row[cn] for cn in ['fSTN', 'tSTN']]
            count = eval(row['count'])
            if (fSTN, tSTN) in flow_oridest:
                flows[loc_lid[fSTN], loc_lid[tSTN]] = count
    assert len(flows) == len(flow_oridest)
    tasks = []
    for i, (loc0, loc1) in enumerate(task_ppdp):
        tasks.append((i, loc_lid[loc0], loc_lid[loc1]))
    #
    problem = [problemName,
               flows, tasks,
               numBundles, minTB, maxTB,
               numLocs, travel_time, thDetour,
               minWS]
    with open(opath.join(pkl_dir, 'problem_%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(problem, fp)
    prmt = convert_prob2prmt(*problem)
    with open(opath.join(pkl_dir, 'prmts_%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(prmt, fp)
    vizInputs = [flow_oridest, task_ppdp]
    with open(opath.join(pkl_dir, 'dplym_%s.pkl' % problemName), 'wb') as fp:
        pickle.dump(vizInputs, fp)
    #
    return prmt


def mrtS2(pkl_dir='_temp'):
    stationSel = '5out'
    stations = STATIONS[stationSel]
    numTasks = 50
    min_durPD = 30
    detourPER, flowPER = PER50, PER75
    minTB, maxTB = 2, 4
    numBundles = int(numTasks / ((minTB + maxTB) / 2)) + 1
    problemName = '%s-nt%d-mDP%d-mTB%d-dp%d-fp%d' % (stationSel, numTasks, min_durPD, maxTB, detourPER, flowPER)
    #
    flow_oridest, task_ppdp, \
    flows, tasks, \
    numLocs, travel_time, thDetour, \
    minWS = gen_instance(stations, numTasks, min_durPD, detourPER, flowPER)
    problem = [problemName,
               flows, tasks,
               numBundles, minTB, maxTB,
               numLocs, travel_time, thDetour,
               minWS]
    prmt = inputConvertPickle(problem, flow_oridest, task_ppdp, pkl_dir)
    #
    return prmt


if __name__ == '__main__':
    # mrtS1()
    mrtS2()
