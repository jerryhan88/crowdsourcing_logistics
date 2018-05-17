import os.path as opath
import os
import pickle, csv
import folium, json, webbrowser
import pandas as pd
from pykml import parser
from shapely.geometry import Polygon, Point
import googlemaps
#
from __path_organizer import ef_dpath, pf_dpath, ez_dpath, mrtNet_dpath, viz_dpath
from sgDistrict import get_distPoly, get_distCBD
from sgDistrict import gen_distCBDJSON, gen_distXCBDJSON

aDayMorning_EZ_fpath = opath.join(ez_dpath, 'EZ-MRT-D20130801-H06H11.csv')
aDayNight_EZ_fpath = opath.join(ez_dpath, 'EZ-MRT-D20130801-H18H23.csv')


SEC60 = 60


def get_coordMRT():
    csv_fpath = opath.join(pf_dpath, 'MRT_coords.csv')
    pkl_fpath = opath.join(pf_dpath, 'MRT_coords.pkl')
    if not opath.exists(csv_fpath):
        with open(csv_fpath, 'w') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_headers = ['Name', 'Lat', 'Lon']
            writer.writerow(new_headers)
        #
        kml_fpath = opath.join(ef_dpath, 'G_MP14_RAIL_STN_PL.kml')
        with open(kml_fpath) as f:
            kml_doc = parser.parse(f).getroot().Document
        mrt_coords = {}
        for pm in kml_doc.Folder.Placemark:
            min_lat, min_lon = 1e400, 1e400
            max_lat, max_lon = -1e400, -1e400
            str_coords = str(pm.MultiGeometry.Polygon.outerBoundaryIs.LinearRing.coordinates)
            for l in ''.join(str_coords.split()).split(',0')[:-1]:
                lon, lat = map(eval, l.split(','))
                if lat < min_lat:
                    min_lat = lat
                if lon < min_lon:
                    min_lon = lon
                if max_lat < lat:
                    max_lat = lat
                if max_lon < lon:
                    max_lon = lon
            cLat, cLon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
            mrt_name = str(pm.name).title()
            if mrt_name == 'Null':
                continue
            for postfix in [' Mrt Station', ' Station', ' Interchange', ' Rail', ' Lrt']:
                if postfix in mrt_name:
                    mrt_name = mrt_name[:-len(postfix)]
                    break
            if mrt_name in ['Outram', 'Tsl', 'Nsle', 'Punggol Central',
                            'Bedok Town Park', 'River Valley', 'Sengkang Central',
                            'Thomson Line', 'Springleaf']:
                continue
            if mrt_name == 'Kallang Bahru':
                mrt_name = 'Geylang Bahru'
            if mrt_name == 'Marymount ':
                mrt_name = 'Marymount'
            #
            if mrt_name in mrt_coords:
                continue
            with open(csv_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([mrt_name, cLat, cLon])
            mrt_coords[mrt_name] = [cLat, cLon]
        #
        with open(pkl_fpath, 'wb') as fp:
            pickle.dump(mrt_coords, fp)
    else:
        with open(pkl_fpath, 'rb') as fp:
            mrt_coords = pickle.load(fp)
    #
    return mrt_coords


def get_districtMRTs():
    csv_fpath = opath.join(pf_dpath, 'DistrictMRTs.csv')
    pkl_fpath = opath.join(pf_dpath, 'DistrictMRTs.pkl')
    if not opath.exists(csv_fpath):
        with open(csv_fpath, 'w') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_headers = ['Name', 'MRTs']
            writer.writerow(new_headers)
        #
        mrt_coords0 = get_coordMRT()
        mrt_coords1 = {}
        for mrt_name, (lat, lon) in mrt_coords0.items():
            mrt_coords1[mrt_name] = Point(lat, lon)
        dist_poly0 = get_distPoly()
        dist_poly1 = {}
        for dist_name, dist_coords in dist_poly0.items():
            dist_poly1[dist_name] = Polygon(dist_coords)
        #
        distMRTs = {}
        for dist_name, distPoly in dist_poly1.items():
            mrts = [mrt_name for mrt_name, mrtPoint in mrt_coords1.items() if mrtPoint.within(distPoly)]
            distMRTs[dist_name] = mrts
            with open(csv_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([dist_name, ';'.join(mrts)])
        #
        with open(pkl_fpath, 'wb') as fp:
            pickle.dump(distMRTs, fp)
    else:
        with open(pkl_fpath, 'rb') as fp:
            distMRTs = pickle.load(fp)
    #
    return distMRTs


def get_mrtLines():
    mrtLines = {}
    for fn in os.listdir(mrtNet_dpath):
        if not fn.endswith('.csv'):
            continue
        lineName = fn[len('Line'):-len('.csv')]
        MRTs = []
        with open(opath.join(mrtNet_dpath, fn)) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                MRTs.append(row['STN'])
        mrtLines[lineName] = MRTs
    #
    return mrtLines


def get_inflowCBD():
    csv_fpath = opath.join(pf_dpath, 'inflowCBD.csv')
    pkl_fpath = opath.join(pf_dpath, 'inflowCBD.pkl')
    if not opath.exists(csv_fpath):
        with open(csv_fpath, 'w') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_headers = ['Name', 'count']
            writer.writerow(new_headers)
        #
        distCBD, districtMRTs = get_distCBD(), get_districtMRTs()
        distCBD_MRTs, MRT_district = [], {}
        for distName, MRTs in districtMRTs.items():
            if distName in distCBD:
                distCBD_MRTs += MRTs
                for MRT_name in MRTs:
                    MRT_district[MRT_name] = distName
        #
        df = pd.read_csv(aDayMorning_EZ_fpath)
        df = df[df['tSTN'].isin(distCBD_MRTs)]
        df = df[['fSTN', 'tSTN', 'count']]
        #
        inflowCBD = {}
        for _, tSTN, count in df.values:
            target_district = MRT_district[tSTN]
            if target_district not in inflowCBD:
                inflowCBD[target_district] = 0
            inflowCBD[target_district] += count
        for distName, count in inflowCBD.items():
            with open(csv_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([distName, count])
        with open(pkl_fpath, 'wb') as fp:
            pickle.dump(inflowCBD, fp)
    else:
        with open(pkl_fpath, 'rb') as fp:
            inflowCBD = pickle.load(fp)
    #
    return inflowCBD


def get_outflowCBD():
    csv_fpath = opath.join(pf_dpath, 'outflowCBD.csv')
    pkl_fpath = opath.join(pf_dpath, 'outflowCBD.pkl')
    if not opath.exists(csv_fpath):
        with open(csv_fpath, 'w') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_headers = ['Name', 'count']
            writer.writerow(new_headers)
        #
        distCBD, districtMRTs = get_distCBD(), get_districtMRTs()
        distCBD_MRTs, MRT_district = [], {}
        for distName, MRTs in districtMRTs.items():
            if distName not in distCBD:
                distCBD_MRTs += MRTs
                for MRT_name in MRTs:
                    MRT_district[MRT_name] = distName
        #
        df = pd.read_csv(aDayNight_EZ_fpath)
        df = df[df['fSTN'].isin(distCBD_MRTs)]
        df = df[['fSTN', 'tSTN', 'count']]
        #
        outflowCBD = {}
        for fSTN, _, count in df.values:
            target_district = MRT_district[fSTN]
            if target_district not in outflowCBD:
                outflowCBD[target_district] = 0
            outflowCBD[target_district] += count
        for distName, count in outflowCBD.items():
            with open(csv_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([distName, count])
        with open(pkl_fpath, 'wb') as fp:
            pickle.dump(outflowCBD, fp)
    else:
        with open(pkl_fpath, 'rb') as fp:
            outflowCBD = pickle.load(fp)
    #
    return outflowCBD


def add_MRTs_onMap(map_osm):
    coordMRT, mrtLines = get_coordMRT(), get_mrtLines()
    for lineName, MRTs in mrtLines.items():
        if lineName.startswith('EW'):
            col, nos = 'green', 3
        elif lineName.startswith('NS'):
            col, nos = 'red', 4
        elif lineName.startswith('NE'):
            col, nos = 'purple', 5
        elif lineName.startswith('CC'):
            col, nos = 'orange', 6
        else:
            col, nos = 'gray', 7
        for mrt_name in MRTs:
            if mrt_name == 'Harbour Front':
                mrt_name = 'Harbourfront'
            if mrt_name == 'MacPherson':
                mrt_name = 'Macpherson'
            if mrt_name == 'Jelapang':
                mrt_name = 'Jelepang'
            if mrt_name == 'South VIEW':
                mrt_name = 'South View'
            if mrt_name == 'one-north':
                mrt_name = 'One North'
            if mrt_name == 'Ten Mile Junction':
                continue

            folium.RegularPolygonMarker(
                tuple(coordMRT[mrt_name]),
                popup='%s' % mrt_name,
                fill_color=col,
                number_of_sides=nos,
                radius=3
            ).add_to(map_osm)


def viz_inflowCBD():
    html_fpath = opath.join(viz_dpath, 'inflowCBD.html')
    csv_fpath = opath.join(pf_dpath, 'inflowCBD.csv')
    if not opath.exists(csv_fpath):
        get_inflowCBD()
    df = pd.read_csv(csv_fpath)
    gjson_fpath = opath.join(pf_dpath, 'districtCBD.json')
    if not opath.exists(gjson_fpath):
        gen_distCBDJSON()
    #
    distPoly = get_distPoly()
    max_lon, max_lat = -1e400, -1e400
    min_lon, min_lat = 1e400, 1e400
    for distName, poly_latlon in distPoly.items():
        for lat, lon in poly_latlon:
            if lat < min_lat:
                min_lat = lat
            if lon < min_lon:
                min_lon = lon
            if max_lat < lat:
                max_lat = lat
            if max_lon < lon:
                max_lon = lon
    #
    lngC, latC = (max_lon + min_lon) / 2.0, (max_lat + min_lat) / 2.0
    map_osm = folium.Map(location=[latC, lngC], zoom_start=11)
    map_osm.choropleth(geo_data=gjson_fpath, data=df,
                     columns=('Name', 'count'),
                     key_on='feature.Name',
                     fill_color='BuPu', fill_opacity=0.7, line_opacity=0.2,
                       threshold_scale=[0, 10000, 20000, 40000, 80000])
    add_MRTs_onMap(map_osm)
    map_osm.save(html_fpath)
    #
    html_url = 'file://%s' % (opath.abspath(html_fpath))
    webbrowser.get('safari').open_new(html_url)


def viz_outflowCBD():
    html_fpath = opath.join(viz_dpath, 'outflowCBD.html')
    csv_fpath = opath.join(pf_dpath, 'outflowCBD.csv')
    if not opath.exists(csv_fpath):
        get_outflowCBD()
    df = pd.read_csv(csv_fpath)
    gjson_fpath = opath.join(pf_dpath, 'districtXCBD.json')
    if not opath.exists(gjson_fpath):
        gen_distXCBDJSON()
    #
    distPoly = get_distPoly()
    max_lon, max_lat = -1e400, -1e400
    min_lon, min_lat = 1e400, 1e400
    for distName, poly_latlon in distPoly.items():
        for lat, lon in poly_latlon:
            if lat < min_lat:
                min_lat = lat
            if lon < min_lon:
                min_lon = lon
            if max_lat < lat:
                max_lat = lat
            if max_lon < lon:
                max_lon = lon
    #
    lonC, latC = (max_lon + min_lon) / 2.0, (max_lat + min_lat) / 2.0
    map_osm = folium.Map(location=[latC, lonC], zoom_start=11)
    map_osm.choropleth(geo_data=gjson_fpath, data=df,
                     columns=('Name', 'count'),
                     key_on='feature.Name',
                     fill_color='BuPu', fill_opacity=0.7, line_opacity=0.2,
                       threshold_scale=[0, 10000, 20000, 40000, 80000])
    add_MRTs_onMap(map_osm)
    #
    map_osm.save(html_fpath)
    #
    html_url = 'file://%s' % (opath.abspath(html_fpath))
    webbrowser.get('safari').open_new(html_url)



def get_MRTpairs():
    lineMRTs = get_mrtLines()
    wholeMRTs = []
    for MRTs in lineMRTs.values():
        wholeMRTs += MRTs
    wholeMRTs = list(set(wholeMRTs))
    MRTparis = set()
    for i, MRT0 in enumerate(wholeMRTs):
        for MRT1 in wholeMRTs[i:]:
            if MRT0 == MRT1:
                continue
            MRTparis.add((MRT0, MRT1))
    return MRTparis


def get_travelTimeMRT_googlemap():
    googleKey1 = 'AIzaSyAQYLeLHyJvNVC7uIbHmnvf7x9XC6murmk'
    googleKey2 = 'AIzaSyDCiqj9QQ-lXWGmzxXM0j-Gbeo_BRlsd0g'
    googleKey3 = 'AIzaSyCsrxK4ZuxQAsGYt3RNHLeGfEFHwq-GIEU'
    googleKey4 = 'AIzaSyB2mRWLDgNcAi99A8wGQXqCqecHjihzEa0'
    googleKey = googleKey2
    #
    csv_fpath = opath.join(pf_dpath, 'travelTimeMRT.csv')
    if not opath.exists(csv_fpath):
        with open(csv_fpath, 'w') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_headers = ['fSTN', 'tSTN', 'travelTime']
            writer.writerow(new_headers)
        target_MRTpairs = get_MRTpairs()

    else:
        handled_MRTpairs = set()
        with open(csv_fpath) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                handled_MRTpairs.add((row['fSTN'], row['tSTN']))
        wholeMRTparis = get_MRTpairs()
        target_MRTpairs = wholeMRTparis.difference(handled_MRTpairs)
        re_handled_MRTpairs = set([(MRT1, MRT0) for MRT0, MRT1 in handled_MRTpairs])
        target_MRTpairs = target_MRTpairs.difference(re_handled_MRTpairs)

        #
    print(len(wholeMRTparis), len(target_MRTpairs))
    LRT_stations = ['Senja', 'Jelapang', 'Segar', 'Samudera', 'Pending', 'Fajar', 'Damai', 'Sumang', 'Riviera']
    mrt_coords = get_coordMRT()
    gmaps = googlemaps.Client(key=googleKey)
    for MRT0, MRT1 in target_MRTpairs:
        loc0 = '%s LRT' % MRT0 if MRT0 in LRT_stations else '%s MRT Station' % MRT0
        loc1 = '%s LRT' % MRT1 if MRT1 in LRT_stations else '%s MRT Station' % MRT1
        res = gmaps.distance_matrix(loc0, loc1,
                                    mode="transit", transit_mode='train')
        elements = res['rows'][0]['elements']
        if elements[0]['status'] == 'NOT_FOUND':
            if MRT0 in LRT_stations:
                if MRT0 == 'Jelapang':
                    loc0 = tuple(mrt_coords['Jelepang'])
                else:
                    loc0 = tuple(mrt_coords[MRT0])
            if MRT1 in LRT_stations:
                if MRT1 == 'Jelapang':
                    loc1 = tuple(mrt_coords['Jelepang'])
                else:
                    loc1 = tuple(mrt_coords[MRT1])
            # if MRT0 == 'Harbour Front': MRT0 = 'Harbourfront'
            # if MRT1 == 'Harbour Front': MRT1 = 'Harbourfront'
            # #
            # if MRT0 == 'MacPherson': MRT0 = 'Macpherson'
            # if MRT1 == 'MacPherson': MRT1 = 'Macpherson'
            # #
            # if MRT0 == 'Jelapang': MRT0 = 'Jelepang'
            # if MRT1 == 'Jelapang': MRT1 = 'Jelepang'
            # #
            # if MRT0 == 'South VIEW': MRT0 = 'South View'
            # if MRT1 == 'South VIEW': MRT1 = 'South View'
            #
            #
            res = gmaps.distance_matrix(loc0, loc1,
                                        mode="transit", transit_mode='train')
            elements = res['rows'][0]['elements']
        try:
            duration = elements[0]['duration']['value'] / SEC60
            with open(csv_fpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow([MRT0, MRT1, duration])
        except KeyError:
            print(MRT0, MRT1)


if __name__ == '__main__':
    # get_MRTs()
    # get_districtPopPoly()
    # get_districtMRTs()
    # get_inflowCBD()
    # get_travelTimeMRT_googlemap()
    #
    # viz_inflowCBD()
    viz_outflowCBD()
