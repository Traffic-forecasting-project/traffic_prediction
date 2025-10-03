'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = Utility functions for using coordinates and bounding boxes
'''

import json
import random
import logging
import pandas as pd

## ========================
## Coordinate utilities
## ========================
def load_arrondissement_data(path: str) -> dict:
    """
        [DEPRECATED] Load polygon definitions for each Paris arrondissement from a CSV file

        Args:
            path (str): Path to the CSV file containing 'arrondissement' and 'polygon' columns

        Returns:
            dict: Dictionary mapping arrondissement numbers to lists of [lon, lat] points
    """
    
    ## Read CSV containing arrondissement and polygon columns
    df = pd.read_csv(path)
    arrondissement_polygons = {}

    ## Iterate over rows to extract polygon data
    for _, row in df.iterrows():
    
        arr = int(row["arrondissement"])  ## Convert arrondissement to int
        points = json.loads(row["polygon"])  ## Parse polygon string into list of points
        arrondissement_polygons[arr] = points  ## Store mapping in dictionary

    return arrondissement_polygons

def load_arrondissement_polygons(path):
    """
        Loads the polygon coordinates of each Paris arrondissement from a CSV file

        Args:
            path (str): Path to the CSV file containing arrondissement boundaries

        Returns:
            dict: A dictionary mapping arrondissement number to its polygon coordinates
    """
    
    ## Load the CSV containing arrondissement numbers and their coordinates
    df = pd.read_csv(path)

    ## Initialize empty dictionary for mapping arrondissement to coordinates
    arr_dict = {}

    ## Iterate over each row to extract and parse the coordinates
    for _, row in df.iterrows():
        num = row["num"]  ## Arrondissement number
        coords_str = row["coordinates"]  ## Coordinate string (likely JSON)

        try:
            coords = json.loads(coords_str)  ## Try parsing as JSON
        except Exception:
            coords = eval(coords_str)  ## Fallback to eval if JSON parsing fails

        arr_dict[num] = coords  ## Store in dictionary

    return arr_dict

def extract_point_list_from_geometry(geometry_str: str) -> list:
    """
        Convert a GeoJSON-like geometry string into a list of [lon, lat] coordinates
        
        Args:
            geometry_str (str): A string representation of the geometry (e.g., GeoJSON Polygon)
        
        Returns:
            list: A list of [lon, lat] coordinate pairs
    """
    
    try:
        geometry = json.loads(geometry_str)
        return geometry.get("coordinates", [])[0]  # Assuming a single polygon
    except Exception as e:
        logging.warning(f"Could not parse geometry: {e}")
        return []

def point_inside_polygon(x, y, polygon):
    """
        Check whether a point is inside a polygon using the ray casting algorithm

        Args:
            x (float): Longitude of the point
            y (float): Latitude of the point
            polygon (list): List of [lon, lat] pairs defining the polygon

        Returns:
            bool: True if the point is inside the polygon, otherwise False
    """
    
    n = len(polygon)
    inside = False

    ## Start with the first point of the polygon
    p1x, p1y = polygon[0]

    ## Iterate over each edge in the polygon
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]  ## Loop back to the start at the end

        ## Check if the horizontal ray crosses the edge vertically
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):

                    ## Compute the x-intersection with the edge
                    if p1y != p2y:
                        xinters = ((y - p1y) * (p2x - p1x)) / (p2y - p1y + 1e-9) + p1x
                    else:
                        xinters = p1x

                    ## Flip 'inside' if the point is to the left of the edge
                    if p1x == p2x or x <= xinters:
                        inside = not inside

        ## Move to next edge
        p1x, p1y = p2x, p2y

    return inside

def get_coordinates(polygon, nb_points):
    """
        Generate random coordinates that lie within a given polygon

        Args:
            polygon (list): List of [lon, lat] pairs forming the polygon boundary
            nb_points (int): Number of valid random points to generate inside the polygon

        Returns:
            list: List of (lon, lat) tuples contained in the polygon
    """
    
    ## Compute bounding box around the polygon
    longitudes = [p[0] for p in polygon]
    latitudes = [p[1] for p in polygon]

    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)

    coordinates = []

    ## Randomly sample points within the bounding box until they fall inside the polygon
    while len(coordinates) < nb_points:
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)

        if point_inside_polygon(lon, lat, polygon):
            coordinates.append((lon, lat))

    return coordinates

def split_bbox(lat1, lon1, lat2, lon2, count):
    """
        Split a bounding box into smaller sub-bounding boxes

        Args:
            lat1 (float): South latitude of the original bbox
            lon1 (float): West longitude of the original bbox
            lat2 (float): North latitude of the original bbox
            lon2 (float): East longitude of the original bbox
            count (int): Number of parts to split each side into (1 = no split)

        Returns:
            list: List of sub-bounding boxes as (lat1, lon1, lat2, lon2) tuples
    """

    ## If no split is requested, return the full bounding box
    if count <= 1:
        return [(lat1, lon1, lat2, lon2)]

    ## Compute latitude and longitude step sizes
    lat_step = (lat2 - lat1) / count
    lon_step = (lon2 - lon1) / count

    bboxes = []

    ## Loop to generate each sub-bounding box
    for i in range(count):
        for j in range(count):
            sub_lat1 = lat1 + i * lat_step
            sub_lat2 = sub_lat1 + lat_step
            sub_lon1 = lon1 + j * lon_step
            sub_lon2 = sub_lon1 + lon_step
            bboxes.append((sub_lat1, sub_lon1, sub_lat2, sub_lon2))

    return bboxes

def get_bbox_from_coords(coords):
    """
        Get the bounding box (min/max lat/lon) from a list of coordinates

        Args:
            coords (list): List of [lon, lat] pairs.

        Returns:
            tuple: (min_lat, min_lon, max_lat, max_lon)
    """

    ## Extract all longitude and latitude values from the coordinate list
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]

    ## Return the bounding box: (min_latitude, min_longitude, max_latitude, max_longitude)
    return min(lats), min(lons), max(lats), max(lons)