# GeoSense Agent 2.0 — Phase I Data Inventory

**Project:** GeoSense Agent 2.0 — Site Suitability Scoring
**Study area:** Kolkata, West Bengal, India
**Study area bounding box:** `(88.30, 22.45, 88.42, 22.67)` — min_lon, min_lat, max_lon, max_lat
(Central Kolkata. The western edge at 88.30°E lies west of the Hooghly for part of its
length, so the area includes the eastern portion of Howrah — Shibpur and the Howrah station
district — as well as the Kolkata Municipal Corporation core. At 0.005° spacing this yields
25 × 45 = 1,125 candidate points over roughly 12.3 km east–west by 24.3 km north–south.)

**Projected CRS for metric calculations:** EPSG:32645 (WGS 84 / UTM zone 45N)
**Study area corners in EPSG:32645 (metres):**

| Corner | Geographic (EPSG:4326) | Easting (m) | Northing (m) |
|---|---|---|---|
| SW | 88.30 E, 22.45 N | 633,767.8 | 2,483,216.6 |
| SE | 88.42 E, 22.45 N | 646,117.3 | 2,483,328.6 |
| NW | 88.30 E, 22.67 N | 633,555.8 | 2,507,573.7 |
| NE | 88.42 E, 22.67 N | 645,885.7 | 2,507,686.5 |

Axis-aligned bounding box in UTM: **633,555.8 – 646,117.3 E, 2,483,216.6 – 2,507,686.5 N**
(12,561.6 m east–west × 24,469.9 m north–south).

The four corners do not form a rectangle. The meridian 88.30 E projects to easting 633,767.8
at the southern edge but 633,555.8 at the northern edge — the same line of longitude moves
212 m west in UTM coordinates over 24 km of northing, because meridians converge towards the
pole while UTM eastings are measured from a fixed central meridian at 87 E. A geographic
rectangle is therefore a slightly tapered, slightly rotated quadrilateral once projected.
The axis-aligned box above is the smallest UTM rectangle enclosing it, and is what should be
used when a rectangular grid is required in projected space.

**Grid spacing.** The grid uses 0.005° spacing (approximately 500 metres). At 22.56 N this is
553.6 m north–south but only 514.2 m east–west, so the cells are 7.7% taller than they are
wide. A grid generated in degrees is not a square grid, and any per-cell area or density
figure derived from it must account for this.

**Grid scale factor.** UTM zone 45N uses a central-meridian scale factor of 0.9996; across
this study area the point scale factor runs from 0.999819 to 0.999862, roughly −160 ppm.
Measured distances are therefore about 0.16 m short per kilometre — negligible relative to
the ~30 m resolution of the input data, but it is why UTM distances are called grid
distances rather than ground distances.

**Maintained by:** Anubhav Ghosh, M.Tech Geoinformatics, Department of Geography, University of Madras
**Last updated:** 19 August 2026

---

## 1. Datasets acquired

### 1.1 OpenStreetMap vector data

| Field | Value |
|---|---|
| Source | Geofabrik Download Server — India / Eastern Zone |
| Source URL | https://download.geofabrik.de/asia/india/eastern-zone.html |
| File downloaded | `eastern-zone-260818-free.shp.zip` (dated snapshot of 18 Aug 2026) |
| Date downloaded | 18 August 2026 (extract dated 18 Aug 2026); organised into the project 19 Aug 2026 |
| Format | ESRI Shapefile (zipped archive) |
| Approximate size | 562 MB compressed / 1.538 GB extracted (101 files across 20 layers) |
| Coordinate system | EPSG:4326 (WGS 84 geographic) |
| Destination folder | `data/shapefiles/osm/` |
| Licence | Open Database Licence (ODbL) — © OpenStreetMap contributors |
| Update frequency | Rebuilt daily by Geofabrik from the OSM planet file |

**Note on version pinning.** The dated snapshot was downloaded in preference to
`eastern-zone-latest-free.shp.zip`. The `latest` file is overwritten every day, so a result
produced from it cannot be reproduced from the same URL afterwards; the dated file names the
exact extract used and keeps the analysis reproducible for as long as Geofabrik retains it.

**Layers used in Phase I**

| Shapefile | Geometry | Represents |
|---|---|---|
| `gis_osm_roads_free_1.shp` | Line | Road and street network, classified by highway type |
| `gis_osm_buildings_a_free_1.shp` | Polygon | Building footprints — note the `_a_` in the name |
| `gis_osm_pois_free_1.shp` | Point | Points of interest mapped as single nodes — shops, banks, small clinics |
| `gis_osm_pois_a_free_1.shp` | Polygon | Points of interest mapped as areas — hospitals, malls, campuses |
| `gis_osm_landuse_a_free_1.shp` | Polygon | Land-use classes — residential, commercial, industrial, forest |
| `gis_osm_places_free_1.shp` | Point | Named settlements and localities |
| `gis_osm_transport_free_1.shp` | Point | Bus stops, railway stations, transport nodes |

**Note on coverage.** Geofabrik does not publish per-state extracts for India; the country is
subdivided into six zones (Central, Eastern, North-Eastern, Northern, Southern, Western).
The Eastern Zone follows the Eastern Zonal Council grouping — Bihar, Jharkhand, Odisha and
West Bengal — and is the smallest published unit containing Kolkata.

A whole-India shapefile package does not exist: the Geofabrik India page states that
`india-latest-free.shp.zip` "is not available for this region; try one of the sub-regions".
The zone extract is therefore the only route to shapefile data, not merely the lighter one.

**Note on point versus area layers.** Nine of the twenty layers in the extract exist in two
forms: a point/line version and an `_a` area version. This is not duplication — OpenStreetMap
contributors may map the same kind of feature either as a single node or as a closed way, and
Geofabrik writes each to a different file. It matters for feature engineering: a large hospital
or shopping mall is almost always mapped as a building polygon and therefore appears only in
`gis_osm_pois_a_free_1.shp`, while a small clinic or ATM appears only in
`gis_osm_pois_free_1.shp`. A "distance to nearest hospital" feature built from the point
file alone would systematically overstate distances by ignoring precisely the largest and
most significant facilities. Both files are therefore read and their geometries combined —
taking the centroid of each area feature — before any nearest-neighbour distance is computed.

Sizes of the layers that dominate the extract: `roads` 471 MB geometry + 266 MB attributes;
`buildings_a` 278 MB geometry + 323 MB attributes; `landuse_a` 53 MB; `water_a` 40 MB;
`waterways` 26 MB; `adminareas_a` 25 MB. The remaining layers are under 5 MB each. Note that
for the buildings layer the attribute table is larger than the geometry itself.

The extract also contains a `README` file from Geofabrik stating the extract date, the layer
schema and the ODbL licence terms; it is retained alongside the data as provenance.

Full layer listing verified in the downloaded extract:
`adminareas_a`, `buildings_a`, `landuse_a`, `natural` + `natural_a`, `places` + `places_a`,
`pofw` + `pofw_a`, `pois` + `pois_a`, `protected_areas_a`, `railways`, `roads`,
`traffic` + `traffic_a`, `transport` + `transport_a`, `waterways`, `water_a` — 20 files.

**Note on the buildings layer name.** Geofabrik writes any feature mapped as an area to a
layer carrying an `_a` suffix, so the building footprints arrive as
`gis_osm_buildings_a_free_1.shp` — verified against the downloaded extract on 19 Aug 2026:
the directory listing contains `gis_osm_buildings_a_free_1.shp` and no unsuffixed variant.
The suffix must be included wherever the layer is referenced in ingestion code.

---

### 1.2 Digital Elevation Model (SRTM)

| Field | Value |
|---|---|
| Source | SRTM GL1 v3 (NASA/USGS), distributed by OpenTopography |
| Source URL | https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1 |
| Subset requested | 88.28–88.44 E, 22.43–22.69 N (study area padded by 0.02°, ≈475 km²) |
| Date downloaded | 19 August 2026 (OpenTopography subset job) |
| Format | GeoTIFF, single band, ≈576 × 936 pixels |
| File | `srtm_gl1_kolkata_30m.tif` (renamed from `output_SRTMGL1.tif` on download) |
| Spatial resolution | 1 arc-second ≈ 30 m at the equator |
| Vertical datum | EGM96 geoid, metres |
| Coordinate system | EPSG:4326 (WGS 84 geographic) |
| Destination folder | `data/raw/elevation/` |
| Licence | Public domain (NASA open data policy) |

**Observed elevation range over the study area:** **−11 m to +35 m** (QGIS band statistics)

**Interpretation of that range.** The 46 m spread is not terrain. Central Kolkata has perhaps
10 m of genuine relief, so both extremes are artefacts of what SRTM actually measures:

- SRTM is a *surface* model, not a *terrain* model. The C-band radar records the height of the
  first surface it reflects from, which over a dense city is rooftops rather than ground. The
  +35 m maximum is almost certainly multi-storey building stock, not high ground.
- Negative values arise chiefly over water. A smooth water surface reflects the radar pulse
  away from the sensor rather than back to it, leaving noisy or interpolated returns; the
  Hooghly and the wetland ponds are the likely source of the −11 m minimum.

Both effects are visible in the rendered raster. The high (orange–red) pixels form a compact
cluster over the dense central business district around Burrabazar and Central Avenue rather
than a smooth topographic rise, which is the signature of building returns and not of terrain.
The Hooghly channel and the East Kolkata Wetlands render as continuous low (green) areas.
Outside the built-up core the surface is close to uniform, consistent with a deltaic plain.

Consequently, the ground elevation of a candidate site cannot be read directly from this
raster in built-up areas. Where a bare-earth surface is required, either a DTM product should
be substituted or the raster should be filtered — for example by taking a low percentile of
values within a moving window — to approximate ground level beneath the roof returns.

**Limitation to note.** Kolkata sits on the Ganges delta and has roughly 10 m of total
relief across the whole study area. SRTM's specified absolute vertical accuracy is 16 m at
90% confidence (measured performance is usually better, of the order of 5–9 m RMSE), which
is comparable to — or larger than — the elevation signal being measured. Local *relative*
differences between neighbouring cells remain informative, but absolute heights should not
be treated as survey-grade, and elevation alone is a weak flood predictor here: waterlogging
in Kolkata is governed largely by drainage capacity and tidal levels in the Hooghly.

---

### 1.3 Google Earth Engine access

| Field | Value |
|---|---|
| Service | Google Earth Engine (noncommercial / academic tier) |
| Registration URL | https://console.cloud.google.com/earth-engine |
| Google Cloud project ID | `ee-anubg2000` (display name "Earth Engine Default Project") |
| Registration type | Noncommercial — academic research |
| Compute tier | Community Tier (free EECU quota; usage 0% at registration) |
| Date registered | 19 August 2026 |
| Status | Registered for noncommercial use; Community Tier quota; access immediate |
| Purpose | Landsat 8/9 and Sentinel-2 imagery access in Phase 2 |

**Note on the registration process.** Earth Engine access is obtained by registering a
Google Cloud project at console.cloud.google.com/earth-engine with a noncommercial
eligibility declaration; access is granted immediately, with no approval wait. The
registration is attached to a Cloud project rather than to a user account, so the project ID
above — not a username — is what the Python client must be given when authenticating in
Phase 2.
---

## 2. Datasets identified for later acquisition

These are required data layers for the full project but are not needed to
train the Phase I baseline model. They are recorded here so the inventory stays complete.

| Data layer | Represents | Intended source | Status |
|---|---|---|---|
| Census demographics | Population density, households, urban/rural class | censusindia.gov.in / data.gov.in | Not yet acquired |
| Flood risk zones | High / medium / low flood risk classification | ndma.gov.in / opendata.gov.in | Not yet acquired |
| Land use / zoning | Residential, commercial, forest, agricultural, industrial | bhuvan.nrsc.gov.in (NLSOF) | Partially met by OSM landuse |
| Climate risk data | Temperature anomaly, drought index, heat island | NASA GIBS API / NOAA Climate Portal | Not yet acquired |

---

## 3. Data handling rules for this project

1. **`data/raw/` is immutable.** Downloaded files are never edited, reprojected or
   overwritten in place. Every transformation writes a new file into `data/processed/`.
   This keeps the pipeline reproducible: if a processing step is found to be wrong, the
   original input is still available to re-run it against.

2. **Provenance is recorded before processing begins.** Source URL and download date are
   written into this file at the moment of download, because a dataset rebuilt daily
   (such as the Geofabrik extract) cannot be reproduced later from the URL alone.

3. **Coordinate systems are converted explicitly.** All source data here is EPSG:4326
   geographic. Distance and area calculations are performed after reprojection to a
   metric CRS — EPSG:32645 (WGS 84 / UTM zone 45N) for the Kolkata study area.

4. **Nothing in `data/` is committed to version control.** `data/raw/` is listed in
   `.gitignore`; the files are large and are reproducible from the sources documented here.

---

## 4. Verification performed

| Dataset | Verification | Result |
|---|---|---|
| OSM roads | Loaded in QGIS over OSM Standard basemap at 1:50 000, centred 22.57°N 88.36°E; road network aligns with the basemap on both banks of the Hooghly | Verified 19 Aug 2026 |
| OSM buildings | Not yet loaded — 278 MB geometry + 323 MB attributes; deferred to Exercise 4, where it is clipped to the study area during PostGIS ingestion rather than rendered whole | Pending (Ex. 4) |
| OSM POIs (area) | `gis_osm_pois_a_free_1` rendered over the basemap; large facilities — Maidan/Fort William, hospitals, campuses — visible as polygons, confirming that significant POIs reside in the area file rather than the point file | Verified 19 Aug 2026 |
| Study-area extent | Earth Engine test rectangle and QGIS views together confirm coverage of the full bounding box on both banks of the Hooghly | Verified 19 Aug 2026 |
| SRTM DEM | Loaded in QGIS, singleband pseudocolor stretched to true statistics; range −11 m to +35 m, extremes attributable to rooftop returns and water-surface noise rather than terrain | Verified 19 Aug 2026 |
| Google Earth Engine | Code Editor test drew the study-area rectangle over Kolkata and returned a geodesic area of 301.44 km², matching the planar estimate from the UTM corners (~301 km²) | Verified 19 Aug 2026 |
