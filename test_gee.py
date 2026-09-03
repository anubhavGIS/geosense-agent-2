# test_gee.py — GeoSense Agent 2.0, Phase 2 / Exercise 1 (Part 3)
# Purpose: confirm that the Earth Engine Python API is authenticated and can
#          query Sentinel-2 Surface Reflectance imagery for the Kolkata study area.
import ee

PROJECT = 'ee-anubg2000'            # Google Cloud project registered for Earth Engine
ee.Initialize(project=PROJECT)

# Kolkata city centre (lon, lat) — inside the study-area bounding box
point = ee.Geometry.Point([88.3639, 22.5726])

collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(point)
              .filterDate('2023-01-01', '2023-12-31')
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
              .sort('CLOUDY_PIXEL_PERCENTAGE'))

print('Images found (2023, < 10 % cloud):', collection.size().getInfo())

image = collection.first()
print('Image ID :', image.id().getInfo())
print('Acquired :', ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo())
print('Cloud %  :', image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo())
print('MGRS tile:', image.get('MGRS_TILE').getInfo())
print('Bands    :', image.bandNames().getInfo())
print('GEE connection working!')