from init_project import *
#
import os.path as opath
import os
import numpy as np
import csv
import time
import networkx as nx
import pandas as pd
import folium
import webbrowser

year, month = 2013, 8
time_slots = [(6, 11), (12, 17), (18, 23)]
topK, MIN_DISTANCE = 25, 8
xStations = ['Bukit Brown', 'Teck Lee']


def flow_MD():
    ftD, days, ftSTNs = {}, set(), set()
    pair_distance = {}
    otime = time.time()
    for fn in ['2013_8_1.csv', '2013_8_2.csv']:
        ifpath = opath.join(dpath['raw'], fn)
        with open(ifpath) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                if row['TRAVEL_MODE'] == 'BUS':
                    continue
                year1, month1, day = map(int, row['RIDE_START_DATE'].split('-'))
                days.add(day)
                assert year1 == year and month1 == month
                hour, _, _ = map(int, row['RIDE_START_TIME'].split(':'))
                for ft, tt in time_slots:
                    if ft <= hour <= tt:
                        if (day, ft, tt) not in ftD:
                            ftD[day, ft, tt] = {}
                        break
                else:
                    continue
                fSTN, tSTN = row['BOARDING_STOP_STN'][len('STN '):], row['ALIGHTING_STOP_STN'][len('STN '):]
                ftSTNs.add((fSTN, tSTN))
                if (fSTN, tSTN) not in ftD[day, ft, tt]:
                    ftD[day, ft, tt][fSTN, tSTN] = []
                try:
                    ftD[day, ft, tt][fSTN, tSTN].append(float(row['RIDE_TIME']))
                    if (fSTN, tSTN) not in pair_distance:
                        pair_distance[fSTN, tSTN] = float(row['RIDE_DISTANCE'])
                except KeyError:
                    pass
                ctime = time.time()
                if (ctime - otime) > 60:
                    print(time.ctime(), day, ft, tt)
                    otime = ctime
    #
    m_ofpath = opath.join(dpath['flow'], 'flow-M%d%02d.csv' % (year, month))
    with open(m_ofpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        new_header = ['fSTN', 'tSTN', 'count', 'distance', 'medianD', 'avgD', 'stdD', 'minD', 'maxD']
        writer.writerow(new_header)
    for fSTN, tSTN in ftSTNs:
        mD = []
        for day in days:
            for ft, tt in time_slots:
                if (day, ft, tt) not in ftD:
                    continue
                d_ofpath = opath.join(dpath['flow'], 'flow-D%d%02d%02d-H%02dH%02d.csv' % (year, month, day, ft, tt))
                if not opath.exists(d_ofpath):
                    with open(d_ofpath, 'wt') as w_csvfile:
                        writer = csv.writer(w_csvfile, lineterminator='\n')
                        new_header = ['fSTN', 'tSTN', 'count', 'distance', 'medianD', 'avgD', 'stdD', 'minD', 'maxD']
                        writer.writerow(new_header)
                if (fSTN, tSTN) not in ftD[day, ft, tt]:
                    continue
                mD += ftD[day, ft, tt][fSTN, tSTN]
                durations = np.array(ftD[day, ft, tt][fSTN, tSTN])
                with open(d_ofpath, 'a') as w_csvfile:
                    writer = csv.writer(w_csvfile, lineterminator='\n')
                    new_row = [fSTN, tSTN, len(durations), pair_distance[fSTN, tSTN],
                               np.median(durations), durations.mean(), durations.std(), durations.min(), durations.max()]
                    writer.writerow(new_row)
        durations = np.array(mD)
        with open(m_ofpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_row = [fSTN, tSTN, len(durations), pair_distance[fSTN, tSTN],
                       np.median(durations), durations.mean(), durations.std(), durations.min(), durations.max()]
            writer.writerow(new_row)
    #
    with open(m_ofpath, 'a') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        writer.writerow(['Ten Mile','Bukit Panjang',0,0.2,1,1,0,1,1])
        writer.writerow(['Bukit Panjang','Ten Mile',0,0.2,1,1,0,1,1])
        writer.writerow(['Kupang', 'Farmway', 0, 1, 7, 7, 0, 7, 7])
        writer.writerow(['Farmway', 'Kupang', 0, 1, 7, 7, 0, 7, 7])
        writer.writerow(['Thanggam', 'Kupang', 0, 0.7, 7, 7, 0, 7, 7])
        writer.writerow(['Kupang', 'Thanggam', 0, 0.7, 7, 7, 0, 7, 7])
        writer.writerow(['Soo Teck', 'Punggol', 0, 0.7, 9, 9, 0, 9, 9])
        writer.writerow(['Punggol', 'Soo Teck', 0, 0.7, 9, 9, 0, 9, 9])
        writer.writerow(['Sumang', 'Soo Teck', 0, 0.4, 8, 8, 0, 8, 8])
        writer.writerow(['Soo Teck', 'Sumang', 0, 0.4, 8, 8, 0, 8, 8])
        writer.writerow(['Samudera', 'Punggol', 0, 1.5, 12, 12, 0, 12, 12])
        writer.writerow(['Punggol', 'Samudera', 0, 1.5, 12, 12, 0, 12, 12])


def get_mrtNetwork():
    m_ofpath = opath.join(dpath['flow'], 'flow-M%d%02d.csv' % (year, month))
    df = pd.read_csv(m_ofpath)
    pair_distance, pair_duration = {}, {}
    for fSTN, tSTN, duration in df[['fSTN', 'tSTN', 'medianD']].values:
        pair_duration[tSTN, fSTN] = duration
        pair_duration[fSTN, tSTN] = duration
    #
    fpath = opath.join(dpath['geo'], 'mrtNetwork.pkl')
    if not opath.exists(fpath):
        G = nx.Graph()
        for fn in ['LineEW.csv', 'LineNS.csv', 'LineNE.csv', 'LineCC.csv',
                   'LineBP.csv', 'LineSTC.csv', 'LinePTC.csv']:
            with open(opath.join(dpath['geo'], fn)) as r_csvfile:
                reader = csv.DictReader(r_csvfile)
                prev_STN = None
                for row in reader:
                    id0, STN, SpecialLink, _lat, _lon = [row.get(cn) for cn in ['ID', 'STN', 'SpecialLink', 'lat', 'lon']]
                    lat0, lon0 = map(float, [_lat, _lon])
                    if not G.has_node(STN):
                        G.add_node(STN, id=id0, lat=lat0, lon=lon0)
                    #
                    if SpecialLink == '' or SpecialLink == 'E':
                        G.add_edge(STN, prev_STN, weight=pair_duration[STN, prev_STN])
                    elif SpecialLink == 'B':
                        pass
                    else:
                        G.add_edge(STN, SpecialLink, weight=pair_duration[STN, SpecialLink])
                    prev_STN = STN
        nx.write_gpickle(G, fpath)
    else:
        G = nx.read_gpickle(fpath)
    return G


def gen_topK_flow(openBrowser=True):
    csv_fpath = opath.join(dpath['flow'], 'topFlow-%d%02d-k%d-D%.2f.csv' % (year, month, topK, MIN_DISTANCE))
    html_fpath = opath.join(dpath['flow'], 'topFlow-%d%02d-k%d-D%.2f.html' % (year, month, topK, MIN_DISTANCE))
    #
    G = get_mrtNetwork()
    max_lon, max_lat = -1e400, -1e400
    min_lon, min_lat = 1e400, 1e400
    for STN in G.node:
        n = G.node[STN]
        lon, lat = n['lon'], n['lat']
        if max_lon < lon:
            max_lon = lon
        if lon < min_lon:
            min_lon = lon
        if max_lat < lat:
            max_lat = lat
        if lat < min_lat:
            min_lat = lat
    lonC, latC = (max_lon + min_lon) / 2.0, (max_lat + min_lat) / 2.0
    map_osm = folium.Map(location=[latC, lonC], zoom_start=11)
    #
    for STN in G.node:
        id, lon, lat = [G.node[STN].get(coords) for coords in ['id', 'lon', 'lat']]
        if id.startswith('EW') or id.startswith('CG'):
            col, nos = 'green', 3
        elif id.startswith('NS'):
            col, nos = 'red', 4
        elif id.startswith('NE'):
            col, nos = 'purple', 5
        elif id.startswith('CC') or id.startswith('CE'):
            col, nos = 'orange', 6
        else:
            col, nos = 'gray', 7
        #
        folium.RegularPolygonMarker(
            (lat, lon),
            popup='%s (%s)' % (STN, id),
            fill_color=col,
            number_of_sides=nos,
            radius=3
        ).add_to(map_osm)
    #
    with open(csv_fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        new_row = ['fSTN', 'tSTN', 'count', 'path']
        writer.writerow(new_row)
    #
    df = pd.read_csv(opath.join(dpath['flow'], 'flow-M%d%02d.csv' % (year, month)))
    df = df.sort_values(by=['count'], ascending=False)
    df = df[(df['distance'] > MIN_DISTANCE)]
    for fSTN, tSTN, count in df[['fSTN', 'tSTN', 'count']][:topK].values:
        coords, path = [], []
        for STN in nx.shortest_path(G, fSTN, tSTN):
            path.append(STN)
            n = G.node[STN]
            lon, lat = n['lon'], n['lat']
            coords.append((lat, lon))
        with open(csv_fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_row = [fSTN, tSTN, count, list(path)]
            writer.writerow(new_row)
        map_osm.add_children(folium.PolyLine(locations=coords, weight=1.0))
    map_osm.save(html_fpath)
    #
    if openBrowser:
        html_url = 'file://%s' % (opath.abspath(html_fpath))
        webbrowser.get('safari').open_new(html_url)


if __name__ == '__main__':
    gen_topK_flow()
    # flow_MD()
    # print(get_mrtNetwork())
    # print(get_topK_flow(k=10))