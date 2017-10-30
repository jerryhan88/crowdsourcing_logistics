import __init__
from init_project import *
#
from _utils.geoFunctions import get_districts
#
from shapely.geometry import Point
import webbrowser
import folium
import pickle
import csv

def run(path=None):
    ofpath = opath.join(dpath['geo'], 'district_stations.pkl')
    dist_poly = get_districts()
    if opath.exists(ofpath):
        dist_stations = None
        with open(ofpath, 'rb') as fp:
            dist_stations = pickle.load(fp)
    else:
        ifpath = opath.join(dpath['geo'], 'all_stations.csv')
        dist_stations = {k: [] for k in dist_poly.iterkeys()}
        with open(ifpath, 'rb') as r_csvfile:
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
                else:
                    assert False
        with open(ofpath, 'wb') as fp:
            pickle.dump(dist_stations, fp)
    #
    max_lon, max_lat = -1e400, -1e400
    min_lon, min_lat = 1e400, 1e400

    for stations in dist_stations.itervalues():
        for _, lon, lat in stations:
            if max_lon < lon:
                max_lon = lon
            if lon < min_lon:
                min_lon = lon
            if max_lat < lat:
                max_lat = lat
            if lat < min_lat:
                min_lat = lat
    lonC, latC = (max_lon + min_lon) / 2.0, (max_lat + min_lat) / 2.0
    map_osm = folium.Map(location=[latC, lonC], zoom_start=14)
    color_map = ['red', 'green', 'blue', 'orange', 'black', 'purple', 'gray', 'white']
    for i, stations in enumerate(dist_stations.itervalues()):
        for locName, lon, lat in stations:
            folium.Marker((lat, lon),
                          popup=locName,
                          icon=folium.Icon(color=color_map[i])
                          ).add_to(map_osm)
    for poly in dist_poly.itervalues():
        l = [(lat, lon) for lon, lat in poly.boundary.coords]
        map_osm.add_children(folium.PolyLine(locations=l, weight=1.0))
    if not path:
        fpath = 'test.html'
        map_osm.save(fpath)
        html_url = 'file://%s' % (opath.join(opath.dirname(__file__), fpath))
        webbrowser.get('safari').open_new(html_url)
    else:
        map_osm.save(path)


if __name__ == '__main__':
    run()