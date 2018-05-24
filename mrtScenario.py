import os.path as opath
import csv, pickle
#
from __path_organizer import ez_dpath, pf_dpath
from problems import convert_prob2prmt

aDayNight_EZ_fpath = opath.join(ez_dpath, 'EZ-MRT-D20130801-H18H23.csv')


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



def temp():
    travel_time[loc_lid['Raffles Place']][loc_lid['Tampines']]
    
    
    MRTs_tt['Paya Lebar', 'Raffles Place'] + 5.41666666666666
    
    MRTs_tt['Paya Lebar', 'Tampines'] + 5.41666666666666
    
    travel_time[loc_lid['Raffles Place']][loc_lid['Hi-Tech Phone Centre Pte Ltd at 810 Geylang Road, #01-06']]
    travel_time[loc_lid['Hi-Tech Phone Centre Pte Ltd at 810 Geylang Road, #01-06']][loc_lid['Tampines']]

    ('Hi-Tech Phone Centre Pte Ltd at 810 Geylang Road, #01-06', 'POPStation@Bedok Point'),

    ('Alfa Marketing at 1G Cantonment Road, #01-07, Pinnacle @ Duxton', 'POPStation@General Post Office'),


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
    
# MIN_HOP = 8
# def random_generation():


def mrtS2(pkl_dir='_temp'):
    from random import seed, randrange
    seed(0)
    #
    thDetour = 80
    problemName = 'mrtS2_dt%d' % thDetour
    numBundles, minTB, maxTB = 4, 2, 3
    minWS = 0.2


    stations = ['Raffles Place', 'Tampines', 'Yishun', 'Jurong East', 'Choa Chu Kang', 'Bishan']
    numTasks = 20

    flow_oridest = [(stations[i], stations[j])
                    for i in range(len(stations)) for j in range(len(stations)) if i != j]

    tempList = []

    with open(opath.join(pf_dpath, 'tt-MRT-LocationPD.csv')) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            nearestMRT, Location = [row[cn] for cn in ['nearestMRT', 'Location']]
            Duration = eval(row['Duration'])
            if nearestMRT in stations:
                tempList.append(Location)
            # locPD_MRT_tt[Location] = [Duration, nearestMRT]

    print(len(flow_oridest))
    print(len(tempList), tempList)
    task_ppdp = []
    while len(task_ppdp) < numTasks:
        i = randrange(len(tempList))
        j = randrange(len(tempList))
        if i != j:
            task_ppdp.append((tempList[i], tempList[j]))

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



    # print(flow_oridest)







if __name__ == '__main__':
    # mrtS1()
    mrtS2()
