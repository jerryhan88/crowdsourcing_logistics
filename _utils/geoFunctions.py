import __init__
from init_project import *
#
from shapely.geometry import LineString, Polygon
import os.path as opath
from pykml import parser
import csv


def get_sgMainBorder():
    fpath = opath.join(dpath['geo'], 'sgMainBorder_manually.csv')
    sgMainBorder = []
    with open(fpath, 'rb') as r_csvfile:
        reader = csv.reader(r_csvfile)
        header = reader.next()
        hid = {h: i for i, h in enumerate(header)}
        for row in reader:
            lon, lat = map(eval, [row[hid[cn]] for cn in ['longitude', 'latitude']])
            sgMainBorder += [(lon, lat)]
    return sgMainBorder


def get_points():
    points = {}
    fpath = opath.join(dpath['geo'], 'points_for_district_division.kml')
    kml_doc = None
    with open(fpath) as f:
        kml_doc = parser.parse(f).getroot().Document
    for pm in kml_doc.Placemark:
        if pm.name not in ['P%d' % (i + 1) for i in range(4)]:
            continue
        _lon, _lat, _ = str(pm.Point.coordinates).split(',')
        points[pm.name] = map(eval, [_lon, _lat])
    return points


def get_districts():
    points = get_points()
    line13 = LineString([points['P1'], points['P3']])
    line24 = LineString([points['P2'], points['P4']])
    center = line13.intersection(line24)

    sgPoly = Polygon(get_sgMainBorder())
    dist_coords = { 'north': [list(*center.coords), points['P1'], points['P2']],
                    'east' : [list(*center.coords), points['P2'], points['P3']],
                    'center': [list(*center.coords), points['P3'], points['P4']],
                    'west': [list(*center.coords), points['P4'], points['P1']]}
    dist_poly = {distName: Polygon(coords).intersection(sgPoly)
                                                for distName, coords in dist_coords.iteritems()}

    return dist_poly


if __name__ == '__main__':
    get_districts()
