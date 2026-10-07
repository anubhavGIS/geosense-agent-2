# similar_sites.py
# Purpose: the find_similar_sites tool - sites with a profile like a reference location.
# Version of Task 2.1 (Exercise 2): the reference is scored with the Phase I site scorer,
# the list of similar sites is still empty. Task 2.3 adds the candidate search.
from site_scorer import score_location            # src/phase1_ml (path set by the server)

from tools.common import in_study_area, outside_study_area


def find_similar_sites(lat, lon, radius_km=5.0, top_k=5):
    """
    Find candidate sites with a profile similar to a reference location (first version).

    Args:
        lat (float): latitude of the reference location in WGS84 decimal degrees
        lon (float): longitude of the reference location in WGS84 decimal degrees
        radius_km (float): search radius around the reference (not used yet)
        top_k (int): number of similar sites to return (not used yet)

    Returns:
        dict: reference (the Phase I score of the location) and similar_sites (empty)

    Example:
        find_similar_sites(22.5530, 88.3540)['reference']['opportunity_score']
    """
    if not in_study_area(lat, lon):
        return outside_study_area(lat, lon, 'Phase I site scorer',
                                  ('reference', 'similar_sites'))
    return {'reference': score_location(lat, lon), 'similar_sites': [],
            'source': 'placeholder - candidate search not yet implemented'}
