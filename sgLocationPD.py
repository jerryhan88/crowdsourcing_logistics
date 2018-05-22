import os.path as opath
import pandas as pd
import csv, pickle
from collections import namedtuple
from shapely.geometry import Polygon, Point
#
from __path_organizer import ef_dpath, pf_dpath
from sgDistrict import get_distPoly

MAX_DETOUR_DURATION = 10  # min.

new_header = ['typePD', 'Lat', 'Lng', 'nearestMRT', 'Duration', 'Location', 'District']
pdLoc = namedtuple('pdLoc', new_header)


def get_locationPD():
    csv_ofpath = opath.join(pf_dpath, 'LocationPD.csv')
    pkl_ofpath = opath.join(pf_dpath, 'LocationPD.pkl')
    if not opath.exists(csv_ofpath):
        _distPoly = get_distPoly()
        distPoly = {}
        for dist_name, poly in _distPoly.items():
            distPoly[dist_name] = Polygon(poly)
        with open(csv_ofpath, 'w') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow(new_header)
        locationPD = []
        #
        ninja_coord = {}
        csv_fpath = opath.join(ef_dpath, 'NinjaBoxPoint.csv')
        with open(csv_fpath) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                loc, lat, lng = [row[cn] for cn in ['Location', 'Lat', 'Lng']]
                ninja_coord[loc] = list(map(eval, [lat, lng]))
        df = pd.read_csv(opath.join(pf_dpath, 'tt-MRT-NinjaLocations.csv'))
        # df = df[(df['Duration'] <= MAX_DETOUR_DURATION)]
        df['typePD'] = df.apply(lambda row: 'D' if 'Ninja Box' in row['Location'] else 'P', axis=1)
        for loc, nearMRT, duration, _, _, typePD in df.values:
            with open(csv_ofpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                lat, lng = ninja_coord[loc]
                p0 = Point(lat, lng)
                for dist_name, poly in distPoly.items():
                    if p0.within(poly):
                        district = dist_name
                        break
                else:
                    assert False
                writer.writerow([typePD, lat, lng, nearMRT, duration, loc, district])
                locationPD.append(pdLoc(typePD, lat, lng, nearMRT, duration, loc, district))
        #
        pop_coord = {}
        csv_fpath = opath.join(ef_dpath, 'POPStation.csv')
        with open(csv_fpath) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                loc, lat, lng = [row[cn] for cn in ['kioskName', 'Lat', 'Lng']]
                pop_coord[loc] = list(map(eval, [lat, lng]))
        df = pd.read_csv(opath.join(pf_dpath, 'tt-MRT-POPStation.csv'))
        # df = df[(df['Duration'] <= MAX_DETOUR_DURATION)]
        typePD = 'D'
        for loc, nearMRT, duration, _, _ in df.values:
            with open(csv_ofpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                lat, lng = pop_coord[loc]
                p0 = Point(lat, lng)
                for dist_name, poly in distPoly.items():
                    if p0.within(poly):
                        district = dist_name
                        break
                else:
                    assert False
                writer.writerow([typePD, lat, lng, nearMRT, duration, loc, district])
                locationPD.append(pdLoc(typePD, lat, lng, nearMRT, duration, loc, district))
        with open(pkl_ofpath, 'wb') as fp:
            pickle.dump(locationPD, fp)
    else:
        with open(pkl_ofpath, 'rb') as fp:
            locationPD = pickle.load(fp)
    #
    return locationPD


if __name__ == '__main__':
    print(get_locationPD())
