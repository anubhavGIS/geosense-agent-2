# climate.py
# Purpose: the get_climate_risk tool - temperature anomaly and drought risk at a location.
#   Data: ERA5-Land monthly reanalysis (ECMWF / Copernicus, 0.1 degree, land only) in
#   Google Earth Engine. The last 12 months available are compared with the same 12
#   calendar months of every year of the 1991-2020 normal period, averaged over a
#   10 km radius (several 0.1-degree cells, so that coastal points still have land).
#   temperature_anomaly_c  mean 2 m air temperature of the 12 months minus the normal
#   precipitation_z        12-month precipitation total in standard deviations of the
#                          30 normal totals (an SPI-12-like index)
#   drought_index          -z / 2 clipped to 0..1: 0 = normal or wetter, 0.5 = moderate
#                          (z = -1), 0.75 = severe (z = -1.5), 1 = extreme (z <= -2)
from datetime import date

from tools.common import cache_get, cache_put, earth_engine

DATASET = 'ECMWF/ERA5_LAND/MONTHLY_AGGR'
NORMAL = (1991, 2020)
RADIUS_M = 10_000
SCALE_M = 11_132                       # 0.1 degree, the native cell of ERA5-Land
CACHE_DAYS = 7                         # ERA5-Land gains one month per month


def _query(lat, lon):
    """One Earth Engine request: 12-month means/totals now and for the 30 normal years."""
    ee = earth_engine()
    monthly = ee.ImageCollection(DATASET).select(['temperature_2m',
                                                  'total_precipitation_sum'])
    end = ee.Date(monthly.aggregate_max('system:time_start')).advance(1, 'month')

    def twelve_months(end_date):           # mean temperature (K), total precipitation (m)
        months = monthly.filterDate(ee.Date(end_date).advance(-12, 'month'), end_date)
        return (months.select('temperature_2m').mean()
                .addBands(months.select('total_precipitation_sum').sum())
                .rename(['t', 'p']))

    normal = ee.ImageCollection.fromImages(ee.List.sequence(*NORMAL).map(
        lambda year: twelve_months(ee.Date.fromYMD(year, end.get('month'), 1))))
    p_stats = normal.select('p').reduce(
        ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True))
    stack = (twelve_months(end).rename(['t_recent', 'p_recent'])
             .addBands(normal.select('t').mean().rename('t_normal'))
             .addBands(p_stats.rename(['p_normal', 'p_sd'])))
    values = stack.reduceRegion(ee.Reducer.mean(),
                                ee.Geometry.Point([lon, lat]).buffer(RADIUS_M), SCALE_M)
    return ee.Dictionary({'values': values, 'end': end.format('YYYY-MM-dd')}).getInfo()


def _month(d, shift):
    """The first day of the month `shift` months after date d."""
    y, m = divmod(d.year * 12 + d.month - 1 + shift, 12)
    return date(y, m + 1, 1)


def _spi_class(z):
    """Wet/dry class of a standardised precipitation anomaly (SPI classes, McKee 1993)."""
    if z >= 1.0:
        return 'extremely wet' if z >= 2 else 'very wet' if z >= 1.5 else 'moderately wet'
    if z > -1.0:
        return 'near normal'
    return 'moderately dry' if z > -1.5 else 'severely dry' if z > -2 else 'extremely dry'


def _temperature_class(anomaly):
    """Plain-language class of a temperature anomaly in degrees C."""
    if anomaly >= 1.0:
        return 'much warmer than normal'
    if anomaly >= 0.5:
        return 'warmer than normal'
    if anomaly > -0.5:
        return 'near normal'
    return 'cooler than normal' if anomaly > -1.0 else 'much cooler than normal'


def get_climate_risk(lat, lon):
    """
    Temperature anomaly and drought risk for a location from ERA5-Land.

    Args:
        lat (float): latitude in WGS84 decimal degrees
        lon (float): longitude in WGS84 decimal degrees

    Returns:
        dict: temperature_anomaly_c and temperature_class, mean and normal temperature,
              precipitation and normal precipitation (mm), precipitation_anomaly_pct,
              precipitation_z, drought_index (0-1), drought_class, period,
              normal_period, from_cache and source

    Example:
        get_climate_risk(13.0827, 80.2707)['drought_class']
    """
    key = f'era5land_{lat:.2f}_{lon:.2f}'
    data, cached = cache_get(key, CACHE_DAYS), True
    if data is None:
        data, cached = _query(lat, lon), False
        cache_put(key, data)
    v = data['values']
    if v.get('t_recent') is None:
        return {'latitude': lat, 'longitude': lon, 'available': False,
                'reason': 'ERA5-Land has no land cells within 10 km of the point',
                'temperature_anomaly_c': None, 'drought_index': None}
    end = date.fromisoformat(data['end'])            # first day after the 12 months
    anomaly = v['t_recent'] - v['t_normal']
    z = (v['p_recent'] - v['p_normal']) / v['p_sd']
    p_anomaly = 100 * (v['p_recent'] / v['p_normal'] - 1)
    return {'latitude': lat, 'longitude': lon,
            'temperature_anomaly_c': round(anomaly, 2),
            'temperature_class': _temperature_class(anomaly),
            'mean_temperature_c': round(v['t_recent'] - 273.15, 2),
            'normal_temperature_c': round(v['t_normal'] - 273.15, 2),
            'precipitation_mm': round(v['p_recent'] * 1000),
            'normal_precipitation_mm': round(v['p_normal'] * 1000),
            'precipitation_anomaly_pct': round(p_anomaly, 1),
            'precipitation_z': round(z, 2),
            'drought_index': round(min(1.0, max(0.0, -z / 2)), 2),
            'drought_class': _spi_class(z),
            'period': f'{_month(end, -12):%b %Y} - {_month(end, -1):%b %Y}',
            'normal_period': f'{NORMAL[0]}-{NORMAL[1]}, same 12 calendar months',
            'from_cache': cached,
            'source': f'ERA5-Land monthly reanalysis ({DATASET}, 0.1 deg) in Google '
                      f'Earth Engine; mean over a {RADIUS_M // 1000} km radius'}
