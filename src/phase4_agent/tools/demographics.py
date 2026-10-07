# demographics.py
# Purpose: the get_demographics tool - population density and healthcare access.
# Version of Task 2.1 (Exercise 2): fixed example values, as in the blueprint's server.
# Task 2.3 replaces them with WorldPop population and the OSM hospitals of Phase I.


def get_demographics(lat, lon):
    """
    Population density and distance to the nearest hospital (fixed example values).

    Args:
        lat (float): latitude in WGS84 decimal degrees
        lon (float): longitude in WGS84 decimal degrees

    Returns:
        dict: latitude, longitude, population_density_per_km2, nearest_hospital_km,
              source

    Example:
        get_demographics(13.0827, 80.2707)['population_density_per_km2']    # 8200
    """
    return {'latitude': lat, 'longitude': lon,
            'population_density_per_km2': 8200, 'nearest_hospital_km': 1.2,
            'source': 'placeholder - fixed example values'}
