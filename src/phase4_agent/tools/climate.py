# climate.py
# Purpose: the get_climate_risk tool - temperature anomaly and drought risk for a location.
# Version of Task 2.1 (Exercise 2): fixed example values, as in the blueprint's server.
# Task 2.3 replaces them with ERA5-Land climate records.


def get_climate_risk(lat, lon):
    """
    Temperature anomaly and drought risk for a location (fixed example values).

    Args:
        lat (float): latitude in WGS84 decimal degrees
        lon (float): longitude in WGS84 decimal degrees

    Returns:
        dict: latitude, longitude, temperature_anomaly_c, drought_index and the source

    Example:
        get_climate_risk(13.0827, 80.2707)['temperature_anomaly_c']     # 1.4
    """
    return {'latitude': lat, 'longitude': lon,
            'temperature_anomaly_c': 1.4, 'drought_index': 0.3,
            'source': 'placeholder - fixed example values'}
