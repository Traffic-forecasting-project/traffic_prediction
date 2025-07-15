import os
import json
import pandas as pd
import requests

# API key and arrondissement number
TOMTOMKEY = os.environ["TOMTOM_KEY"]
ARR_NUMBER = 7

# Load arrondissement boundaries
csv_path = "src/data/arrondissements.csv"
arr_df = pd.read_csv(csv_path, delimiter=";")

# Extract bounding box from geometry (Polygon only)
def extract_bbox_from_geometry(geometry_str):
    geom = json.loads(geometry_str)
    coords = geom["coordinates"][0]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    return min(lons), min(lats), max(lons), max(lats)

# Request traffic flow data at a specific point
def get_flow_from_point(lat, lon, api_key, zoom=10, unit="KMPH"): 
    url = (
        f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/{zoom}/json"
        f"?key={api_key}&point={lat},{lon}&unit={unit}"
    )
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json()
    else:
        raise RuntimeError(f"Flow request failed: {resp.status_code} {resp.text}")

# Get midpoint of a line (list of coordinates)
def extract_middle_point(coords):
    mid = len(coords) // 2
    return coords[mid][1], coords[mid][0]

# Retrieve incidents for a given arrondissement
def get_incidents_df(arr_number, df, api_key):
    row = df[df["Numéro d’arrondissement"] == arr_number].iloc[0]
    bbox = extract_bbox_from_geometry(row["Geometry"])
    bbox_str = ",".join(map(str, bbox))
    url_inc = (
        f"https://api.tomtom.com/traffic/services/5/incidentDetails?key={api_key}"
        f"&bbox={bbox_str}"
        "&fields={incidents{type,geometry{coordinates},properties{iconCategory,magnitudeOfDelay}}}"
        "&language=fr-FR"
        "&timeValidityFilter=present"
    )
    resp = requests.get(url_inc)
    if resp.status_code != 200:
        raise RuntimeError(f"Incident request failed: {resp.status_code} {resp.text}")
    data = resp.json().get("incidents", [])
    return pd.json_normalize(data)

if __name__ == "__main__":
    print("Using API key:", TOMTOMKEY)

    # Example: traffic flow request for a fixed point in Paris
    center_lat, center_lon = 48.8592, 2.3128
    flow_data = get_flow_from_point(center_lat, center_lon, TOMTOMKEY)
    print("Flow currentSpeed:", flow_data.get("flowSegmentData", {}).get("currentSpeed"))

    # Get traffic incidents for the specified arrondissement
    df_incidents = get_incidents_df(ARR_NUMBER, arr_df, TOMTOMKEY)
    print(f"{len(df_incidents)} incident(s) found in arr {ARR_NUMBER}.")
    print(df_incidents.head())

    # For each incident, query traffic flow around the midpoint
    if not df_incidents.empty:
        for i in range(len(df_incidents)):
            line_incident = df_incidents.loc[i, "geometry.coordinates"]
            lat, lon = extract_middle_point(line_incident)
            incident_flow = get_flow_from_point(lat, lon, TOMTOMKEY)
            current_speed = incident_flow.get("flowSegmentData", {}).get("currentSpeed")
            magnitude = df_incidents.loc[i, "properties.magnitudeOfDelay"]
            icon = df_incidents.loc[i, "properties.iconCategory"]
            print(f"Incident {i}: icon={icon}, magnitude={magnitude}, flow_speed={current_speed} KMPH")
