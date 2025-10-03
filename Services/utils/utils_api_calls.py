'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = Utility functions for live data collection
'''

import logging

from Services.FastAPI.src.constants import WEATHER_KEY, TOMTOM_KEY, SAMPLE_ALL_INCIDENT_POINTS
from Services.utils.utils import convert_to_local_timezone, safe_request

## ========================
## API Call utilities
## ========================
def get_weather(api_key, lat, lon):
    """
        Retrieves current weather data from OpenWeatherMap API for a given location

        Args:
            api_key (str): OpenWeatherMap API key (pass None to use default constant)
            lat (float): Latitude of the location
            lon (float): Longitude of the location

        Returns:
            dict: Dictionary containing temperature, wind speed, and rain volume (if available)
    """
    
    ## OpenWeatherMap endpoint
    url = "https://api.openweathermap.org/data/2.5/weather"

    ## Request parameters
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key or WEATHER_KEY,
        "units": "metric"
    }

    ## Send request and parse response
    r = safe_request(url, params)
    data = r.json()

    ## Extract useful weather data
    temp = data.get("main", {}).get("temp")
    wind = data.get("wind", {}).get("speed")
    rain = data.get("rain", {}).get("1h") if "rain" in data else 0.0

    return {
        "temp": temp,
        "wind": wind,
        "rain": rain
    }

def get_traffic_flow(api_key, lat, lon):
    """
        Retrieves real-time traffic flow data from the TomTom Traffic API for a specific location

        Args:
            api_key (str): TomTom API key (pass None to use default constant)
            lat (float): Latitude of the point
            lon (float): Longitude of the point

        Returns:
            dict: Dictionary containing average speed, free-flow speed, and jam factor
    """
    
    ## TomTom Traffic Flow API endpoint
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

    ## Parameters for the API call
    params = {
        "point": f"{lat},{lon}",
        "key": api_key or TOMTOM_KEY
    }

    ## Perform API request safely
    r = safe_request(url, params)
    data = r.json()

    ## Navigate into the API response
    flow_data = data.get("flowSegmentData", {})

    ## Compute traffic-related metrics
    avg_speed = flow_data.get("currentSpeed")
    free_flow_speed = flow_data.get("freeFlowSpeed")
    jam_factor = (
        flow_data.get("currentTravelTime") / flow_data.get("freeFlowTravelTime")
        if flow_data.get("freeFlowTravelTime") and flow_data.get("currentTravelTime")
        else None
    )

    return {
        "avg_speed": avg_speed,
        "free_flow_speed": free_flow_speed,
        "jam_factor": jam_factor
    }

def get_incidents(api_key, lat1, lon1, lat2, lon2):
    """
        Fetches incident data from TomTom API within a specified bounding box

        Args:
            api_key (str): TomTom API key (pass None to use default constant)
            lat1 (float): Southern latitude of the bounding box
            lon1 (float): Western longitude of the bounding box
            lat2 (float): Northern latitude of the bounding box
            lon2 (float): Eastern longitude of the bounding box

        Returns:
            List[dict]: List of cleaned incident dictionaries containing metadata and coordinates
    """
    
    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"

    params = {
        "key": api_key or TOMTOM_KEY,
        "bbox": f"{lon1},{lat1},{lon2},{lat2}",
        "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code,iconCategory},startTime,endTime,from,to,length,delay,roadNumbers,timeValidity,probabilityOfOccurrence,numberOfReports,lastReportTime,tmc{countryCode,tableNumber,tableVersion,direction,points{location,offset}}}}}",
        "language": "en-GB",
        "timeValidityFilter": "present"
    }

    ## Perform API request safely
    r = safe_request(url, params)
    data = r.json()

    incidents = data.get("incidents", [])
    logging.info(f"{len(incidents)} incidents (uniques) extracted.")

    processed_incidents = []

    for inc in incidents:
        props = inc.get("properties", {})
        coords = inc.get("geometry", {}).get("coordinates", [])

        ## Skip if no coordinates
        if not coords or not isinstance(coords, list):
            continue

        ## Keep only first point if sampling is limited
        if not SAMPLE_ALL_INCIDENT_POINTS:
            coords = coords[0:1]

        ## Iterate over incident points
        for coord in coords:
            if not isinstance(coord, list) or len(coord) != 2:
                continue

            lon_val, lat_val = coord[0], coord[1]

            ## Build structured incident dictionary
            incident_data = {
                "incident_count": 1,
                "incident_magnitudes": [props.get("magnitudeOfDelay")],
                "incident_delays": [props.get("delay")],
                "incident_roads": [props.get("roadNumbers", [])],
                "incident_id": props.get("id"),
                "icon_category": props.get("iconCategory"),
                "start_time": convert_to_local_timezone(props.get("startTime")),
                "end_time": convert_to_local_timezone(props.get("endTime")),
                "from_location": props.get("from"),
                "to_location": props.get("to"),
                "length": props.get("length"),
                "time_validity": props.get("timeValidity"),
                "probability": props.get("probabilityOfOccurrence"),
                "num_reports": props.get("numberOfReports"),
                "last_report": props.get("lastReportTime"),
                "tmc_countryCode": props.get("tmc", {}).get("countryCode") if props.get("tmc") else None,
                "tmc_tableNumber": props.get("tmc", {}).get("tableNumber") if props.get("tmc") else None,
                "tmc_tableVersion": props.get("tmc", {}).get("tableVersion") if props.get("tmc") else None,
                "tmc_direction": props.get("tmc", {}).get("direction") if props.get("tmc") else None,
                "event_descriptions": [e.get("description") for e in props.get("events", []) if "description" in e],
                "incident_coords": coords,
                "lat": lat_val,
                "lon": lon_val
            }

            processed_incidents.append(incident_data)

    return processed_incidents

def get_incidents_per_coordinate(api_key, lat1, lon1, lat2, lon2, ts=None):
    """
        [DEPRECATED] Retrieve granular incident data from TomTom API,
        returning one row per coordinate involved in the incident geometry

        Args:
            api_key (str): TomTom API key (pass None to use default constant)
            lat1 (float): Minimum latitude of the bounding box
            lon1 (float): Minimum longitude of the bounding box
            lat2 (float): Maximum latitude of the bounding box
            lon2 (float): Maximum longitude of the bounding box

        Returns:
            list of dict: One entry per coordinate with detailed incident properties
    """

    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"

    params = {
        "key": api_key or TOMTOM_KEY,
        "bbox": f"{lon1},{lat1},{lon2},{lat2}",
        "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code,iconCategory},startTime,endTime,from,to,length,delay,roadNumbers,timeValidity,probabilityOfOccurrence,numberOfReports,lastReportTime,tmc{countryCode,tableNumber,tableVersion,direction,points{location,offset}}}}}",
        "language": "en-GB",
        "timeValidityFilter": "present"
    }

    ## Request API and parse response
    r = safe_request(url, params)
    data = r.json()

    if not isinstance(data, dict):
        logging.warning("Invalid response from TomTom API.")
        return []

    incidents = data.get("incidents", [])
    result_rows = []

    for inc in incidents:
        geometry = inc.get("geometry", {})
        props = inc.get("properties", {})
        coords = geometry.get("coordinates", [])

        ## Each coordinate becomes its own row
        for coord in coords:
            if isinstance(coord, list) and len(coord) >= 2:
                lat, lon = coord[1], coord[0]

                row = {
                    "lat": lat,
                    "lon": lon,
                    "incident_count": 1,
                    "incident_magnitudes": [props.get("magnitudeOfDelay")],
                    "incident_delays": [props.get("delay")],
                    "incident_roads": [props.get("roadNumbers", [])],
                    "incident_id": props.get("id"),
                    "icon_category": props.get("iconCategory"),
                    "start_time": props.get("startTime"),
                    "end_time": props.get("endTime"),
                    "from_location": props.get("from"),
                    "to_location": props.get("to"),
                    "length": props.get("length"),
                    "time_validity": props.get("timeValidity"),
                    "probability": props.get("probabilityOfOccurrence"),
                    "num_reports": props.get("numberOfReports"),
                    "last_report": props.get("lastReportTime"),
                    "tmc_countryCode": props.get("tmc", {}).get("countryCode") if props.get("tmc") else None,
                    "tmc_tableNumber": props.get("tmc", {}).get("tableNumber") if props.get("tmc") else None,
                    "tmc_tableVersion": props.get("tmc", {}).get("tableVersion") if props.get("tmc") else None,
                    "tmc_direction": props.get("tmc", {}).get("direction") if props.get("tmc") else None,
                    "event_descriptions": [e.get("description") for e in props.get("events", []) if "description" in e],
                }

                result_rows.append(row)

    logging.info(f"Number of incident coordinates found: {len(result_rows)}")
    
    return result_rows