import __init__
from init_project import *
#
from _utils.geoFunctions import get_districts
#
from shapely.geometry import Point
import webbrowser
import folium
import csv

def run():
    fpath = opath.join(dpath['geo'], 'all_stations.csv')
    dist_poly = get_districts()
    dist_stations = {k: [] for k in dist_poly.iterkeys()}
    with open(fpath, 'rb') as r_csvfile:
        reader = csv.reader(r_csvfile)
        header = reader.next()
        hid = {h: i for i, h in enumerate(header)}
        for row in reader:
            locName = row[hid['location_name']]
            lon, lat = map(eval, [row[hid[cn]] for cn in 'lon lat'.split()])
            p = Point(lon, lat)
            for distName, poly in dist_poly.iteritems():
                if p.within(poly):
                    dist_stations[distName] += [[locName, lon, lat]]
                    break
    print dist_stations



# Point(*coordinate)
# return p.within(self)


if __name__ == '__main__':
    run()