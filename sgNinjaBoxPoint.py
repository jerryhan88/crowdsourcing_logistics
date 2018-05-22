import os.path as opath
import os
import csv
import googlemaps
from geopy.distance import vincenty
import folium, webbrowser
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
#
from __path_organizer import ef_dpath, pf_dpath, viz_dpath
from sgMRT import get_distPoly, get_coordMRT
from sgMRT import add_MRTs_onMap

MIN60, SEC60 = 60.0, 60.0
Meter1000 = 1000.0
WALKING_SPEED = 5.0  # km/hour

url = 'https://collect.ninjavan.co/en-sg/mapsearch'
googleKey = 'AIzaSyAQYLeLHyJvNVC7uIbHmnvf7x9XC6murmk'


def crawl_NinjaBoxPoint():
    csv_fpath = opath.join(ef_dpath, 'NinjaBoxPoint.csv')
    with open(csv_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        new_headers = ['Location', 'Postcode', 'Lat', 'Lng']
        writer.writerow(new_headers)
    gmaps = googlemaps.Client(key=googleKey)
    #
    wd = webdriver.Firefox(executable_path=os.getcwd() + '/geckodriver')
    wd.get(url)
    WebDriverWait(wd, 5)
    element = wd.find_element_by_class_name('leaflet-marker-pane')
    for ele in element.find_elements_by_tag_name('img'):
        wd.execute_script("arguments[0].click();", ele)
        WebDriverWait(wd, 1)
        ele_cp = wd.find_element_by_class_name('ng-binding')
        ele_postcode = wd.find_element_by_class_name('address_box').find_element_by_id('dp_postcode')
        res = gmaps.geocode(address=ele_postcode.text, components = {'country': 'Singapore'})
        loc = res[0]['geometry']['location']
        lat, lng = [loc.get(k) for k in ['lat', 'lng']]
        with open(csv_fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow([ele_cp.text, ele_postcode.text, lat, lng])
    wd.quit()


def get_latlngFromAddress():
    add = '196A Bukit Batok Street 22'
    gmaps = googlemaps.Client(key=googleKey)
    res = gmaps.geocode(address=add, components={'country': 'Singapore'})
    loc = res[0]['geometry']['location']
    lat, lng = [loc.get(k) for k in ['lat', 'lng']]
    print('%f,%f' % (lat, lng))



def viz_NinjaBoxPoint():
    html_fpath = opath.join(viz_dpath, 'NinjaBoxPoint.html')
    csv_fpath = opath.join(ef_dpath, 'NinjaBoxPoint.csv')
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

    with open(csv_fpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            loc = row['Location']
            lat, lon = [eval(row[cn]) for cn in ['Lat', 'Lng']]
            if 'Ninja Box' in loc:
                folium.Marker([lat, lon],
                              popup='%s' % loc,
                              icon=folium.Icon(color='red', icon='info-sign')
                              ).add_to(map_osm)
            else:
                folium.Marker([lat, lon]).add_to(map_osm)
    add_MRTs_onMap(map_osm)
    map_osm.save(html_fpath)
    #
    html_url = 'file://%s' % (opath.abspath(html_fpath))
    webbrowser.get('safari').open_new(html_url)


def travelTime_MRT_NinjaLocation():
    ofpath = opath.join(pf_dpath, 'tt-MRT-NinjaLocations.csv')
    with open(ofpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        new_headers = ['Location', 'nearestMRT', 'Duration', 'Distance', 'By']
        writer.writerow(new_headers)
    #
    ifpath = opath.join(ef_dpath, 'NinjaBoxPoint.csv')
    mrt_coords = get_coordMRT()
    gmaps = googlemaps.Client(key=googleKey)
    with open(ifpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            Location = row['Location']
            locLat, locLng = [eval(row[cn]) for cn in ['Lat', 'Lng']]
            min_dist2, nearestMRT = 1e400, None
            for mrtName, (mrtLat, mrtLng) in mrt_coords.items():
                dist2 = (mrtLat - locLat) ** 2 + (mrtLng - locLng) ** 2
                if dist2 < min_dist2:
                    min_dist2, nearestMRT = dist2, mrtName
            res = gmaps.distance_matrix((locLat, locLng), tuple(mrt_coords[nearestMRT]),
                                        mode="walking")
            elements = res['rows'][0]['elements'][0]
            distance, duration, status = [elements.get(k) for k in ['distance', 'duration', 'status']]
            new_row = [Location, nearestMRT]
            if status == 'OK':
                new_row += [duration['value'] / SEC60, distance['value'] / Meter1000, 'googlemap']
            else:
                distance = vincenty((locLat, locLng), tuple(mrt_coords[nearestMRT])).km
                duration = (distance / WALKING_SPEED) * MIN60
                new_row += [duration, distance, 'vincenty']
            with open(ofpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow(new_row)


if __name__ == '__main__':
    # run()
    travelTime_MRT_NinjaLocation()