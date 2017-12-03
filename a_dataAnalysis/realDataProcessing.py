from init_project import *
#
import csv

ifpath = opath.join(dpath['raw'], '2013_8_1.csv')
time_slots = [(6, 11), (12, 17), (18, 23)]


with open(ifpath) as r_csvfile:
    reader = csv.DictReader(r_csvfile)
    for row in reader:
        if row['TRAVEL_MODE'] == 'BUS':
            continue
        year, month, day = map(int, row['RIDE_START_DATE'].split('-'))
        if day != 3:
            continue
        hour, minute, sec = map(int, row['RIDE_START_TIME'].split(':'))
        for ft, tt in time_slots:
            if ft <= hour <= tt:
                ofpath = opath.join(dpath['dayInstance'], 'dayInstance-%d%02d%02d-%02d-%02d.csv' % (year, month, day,
                                                                                                    ft, tt))
                break
        else:
            continue
        if not opath.exists(ofpath):
            with open(ofpath, 'wt') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                new_header = ['rideStartTime', 'fromSTN', 'toSTN', 'duration']
                writer.writerow(new_header)
        fsName, tsName = row['BOARDING_STOP_STN'][len('STN '):], row['ALIGHTING_STOP_STN'][len('STN '):]
        try:
            with open(ofpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([row['RIDE_START_TIME'],
                                 fsName, tsName, row['RIDE_TIME']])
        except KeyError:
            pass