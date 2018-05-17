import os.path as opath
import os
import csv
import googlemaps
from geopy.distance import vincenty
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
#
from __path_organizer import ef_dpath, pf_dpath, viz_dpath
from sgMRT import get_coordMRT

MIN60, SEC60 = 60.0, 60.0
Meter1000 = 1000.0
WALKING_SPEED = 5.0  # km/hour

url = 'https://www.mypopstation.com/locations'
googleKey = 'AIzaSyAQYLeLHyJvNVC7uIbHmnvf7x9XC6murmk'

def crawl_POPStarion():
    csv_fpath = opath.join(ef_dpath, 'POPStation.csv')
    with open(csv_fpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        new_headers = ['kioskName', 'Address', 'Lat', 'Lng']
        writer.writerow(new_headers)
    gmaps = googlemaps.Client(key=googleKey)
    #
    wd = webdriver.Firefox(executable_path=os.getcwd() + '/geckodriver')
    wd.get(url)
    WebDriverWait(wd, 5)
    list_element = wd.find_element_by_class_name('locations-list')
    elements = list_element.find_elements_by_class_name('kiosk-block')
    for ele in elements:
        kioskName = ele.find_element_by_class_name('kiosk-name').text
        adds = ele.find_element_by_class_name('location-address-blocker').text    
        res = gmaps.geocode(address=adds, components = {'country': 'Singapore'})
        loc = res[0]['geometry']['location']    
        lat, lng = [loc.get(k) for k in ['lat', 'lng']]
        with open(csv_fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow([kioskName, adds, lat, lng])
    wd.quit()


def travelTime_MRT_POPStation():
    ofpath = opath.join(pf_dpath, 'tt-MRT-POPStation.csv')
    with open(ofpath, 'w') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        new_headers = ['kioskName', 'nearestMRT', 'Duration', 'Distance', 'By']
        writer.writerow(new_headers)
    #
    ifpath = opath.join(ef_dpath, 'POPStation.csv')
    mrt_coords = get_coordMRT()
    gmaps = googlemaps.Client(key=googleKey)
    with open(ifpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            kioskName = row['kioskName']
            adds = row['Address']
            popLat, popLng = [eval(row[cn]) for cn in ['Lat', 'Lng']]
            min_dist2, nearestMRT = 1e400, None
            for mrtName, (mrtLat, mrtLng) in mrt_coords.items():
                dist2 = (mrtLat - popLat) ** 2 + (mrtLng - popLng) ** 2
                if dist2 < min_dist2:
                    min_dist2, nearestMRT = dist2, mrtName
            res = gmaps.distance_matrix(adds, tuple(mrt_coords[nearestMRT]),
                                    mode="walking")
            elements = res['rows'][0]['elements'][0]
            distance, duration, status = [elements.get(k) for k in ['distance', 'duration', 'status']]
            new_row = [kioskName, nearestMRT]
            if status == 'OK':
                new_row += [duration['value'] / SEC60, distance['value'] / Meter1000, 'googlemap']
            else:
                distance = vincenty((popLat, popLng), tuple(mrt_coords[nearestMRT])).km
                duration = (distance / WALKING_SPEED) * MIN60
                new_row += [duration, distance, 'vincenty']
            with open(ofpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                writer.writerow(new_row)
            

if __name__ == '__main__':
    travelTime_MRT_POPStation()