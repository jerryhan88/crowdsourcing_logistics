import __init__
from init_project import *
#
from _utils.geoFunctions import get_districts
#
import pickle
import csv


def run():
    ofpath = opath.join(dpath['geo'], 'district_stations.pkl')
    dist_poly = get_districts()
    if opath.exists(ofpath):
        dist_stations = None
        with open(ofpath, 'rb') as fp:
            dist_stations = pickle.load(fp)
    station_district = {}
    for distName, stations in dist_stations.iteritems():
        for stationName, _, _ in stations:
            station_district[stationName] = distName

    ifpath = opath.join(dpath['raw'], '2013_8_1.csv')
    time_slots = [(6, 11), (12, 17), (18, 23)]
    with open(ifpath, 'rb') as r_csvfile:
        reader = csv.reader(r_csvfile)
        headers = reader.next()
        hid = {h: i for i, h in enumerate(headers)}
        for row in reader:
            if row[hid['TRAVEL_MODE']] == 'BUS':
                continue
            year, month, day = map(int, row[hid['RIDE_START_DATE']].split('-'))
            if day != 3:
                continue
            hour, minute, sec = map(int, row[hid['RIDE_START_TIME']].split(':'))
            for ft, tt in time_slots:
                if ft <= hour <= tt:
                    ofpath = opath.join(dpath['dayInstance'], 'dayInstance-D%02d-%02d-%02d.csv' % (day, ft, tt))
                    break
            else:
                continue
            if not opath.exists(ofpath):
                with open(ofpath, 'wt') as w_csvfile:
                    writer = csv.writer(w_csvfile, lineterminator='\n')
                    new_header = ['rideStartTime', 'fromSTN', 'toSTN', 'fromDistrict', 'toDistrict']
                    writer.writerow(new_header)
            fsName, tsName = row[hid['BOARDING_STOP_STN']][len('STN '):], row[hid['ALIGHTING_STOP_STN']][len('STN '):]
            try:
                with open(ofpath, 'a') as w_csvfile:
                    writer = csv.writer(w_csvfile, lineterminator='\n')
                    writer.writerow([row[hid['RIDE_START_TIME']],
                                     fsName, tsName,
                                     station_district[fsName], station_district[tsName]])
            except KeyError:
                pass
                # print fsName, tsName




if __name__ == '__main__':
    run()