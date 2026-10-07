# mcp_server.py
# Purpose: FastAPI server exposing the GeoSense analysis modules as seven MCP-compatible
#          tools - REST routes /tools/<name> (tested at /docs and in Postman) - and the
#          same seven tools through the Model Context Protocol at /mcp
#          Exercise 5: CORS for the digital twin (5173) and Streamlit (8501), and the
#          agent's latest result at /agent/latest_result (POSTed by the agent, polled
#          by the twin)
# Run from the project root (geosense environment):
#   python src/phase4_agent/mcp_server.py
import inspect
import itertools
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # torch + numpy OpenMP on Windows
import torch     # first: on Windows its DLLs (MKL, OpenMP) must load before shap,
                 # scikit-learn and geopandas bring in other versions of them

sys.path.append('src/phase1_ml')      # relative to the folder Python is started from - the
sys.path.append('src/phase2_dl')      # project root, which the Phase I and III modules
sys.path.append('src/phase3_lidar')   # also need for models/ and data/

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, ValidationError, model_validator
from sqlalchemy.exc import OperationalError

from site_scorer import score_location
from image_classifier import classify_imagery
from lidar_analyzer import analyze_bbox
from streetview_semantic import analyze_location as streetview_analyze

from tools.common import (ToolError, in_study_area, jsonable, outside_study_area, redact,
                          study_area_text)
from tools.climate import get_climate_risk
from tools.demographics import get_demographics
from tools.similar_sites import find_similar_sites

COVERAGE = (f'Coverage: site scoring, land cover and similar sites - the study area '
            f'({study_area_text()}); LiDAR - the Dupont Circle tile, Washington DC; '
            f'street-level imagery, climate and population - worldwide. Outside its '
            f'coverage a tool answers available = false with a reason.')
WRAPPED, CONTEXT = 'Phase I-III modules', 'Context and search tools'
AGENT = 'Agent and digital twin'
CORS_ORIGINS = os.getenv('CORS_ORIGINS',                # browser pages allowed to call us
                         'http://localhost:5173,http://localhost:8501').split(',')
UNAVAILABLE = (ToolError, OperationalError, OSError)    # database, Earth Engine, network
_lock = threading.Lock()          # one analysis at a time: the modules cache large models


@asynccontextmanager
async def lifespan(app):
    """Start and stop the session manager of the /mcp endpoint with the server."""
    async with mcp.session_manager.run():
        yield


app = FastAPI(title='GeoSense Agent MCP Server', version='1.0', lifespan=lifespan,
              description='Seven geospatial analysis tools of GeoSense Agent 2.0, one '
                          'POST route each (JSON in, JSON out, inputs validated by '
                          'Pydantic), also served through the Model Context Protocol at '
                          '/mcp. ' + COVERAGE)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=['*'],
                   allow_headers=['*'])


class LocationInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, examples=[22.5530],
                            description='Latitude in WGS84 decimal degrees')
    longitude: float = Field(..., ge=-180, le=180, examples=[88.3540],
                             description='Longitude in WGS84 decimal degrees')


class BboxInput(BaseModel):
    min_lat: float = Field(..., ge=-90, le=90, examples=[38.9085],
                           description='Southern edge, WGS84 decimal degrees')
    min_lon: float = Field(..., ge=-180, le=180, examples=[-77.0450],
                           description='Western edge, WGS84 decimal degrees')
    max_lat: float = Field(..., ge=-90, le=90, examples=[38.9110],
                           description='Northern edge, WGS84 decimal degrees')
    max_lon: float = Field(..., ge=-180, le=180, examples=[-77.0420],
                           description='Eastern edge, WGS84 decimal degrees')

    @model_validator(mode='after')
    def corners_in_order(self):
        if self.min_lat >= self.max_lat or self.min_lon >= self.max_lon:
            raise ValueError('min_lat must be below max_lat and min_lon below max_lon')
        return self


class SimilarSitesInput(LocationInput):
    radius_km: float = Field(5.0, gt=0, le=25,
                             description='Search radius around the reference, km')
    top_k: int = Field(5, ge=1, le=20, description='Number of similar sites to return')


def run(tool, fn, *args, **kwargs):
    """
    Run one analysis function and return its result as plain JSON.

    Args:
        tool (str): the tool name, used in error messages
        fn (callable): the analysis function; *args and **kwargs are passed to it

    Returns:
        dict: the result with numpy values converted and available = True; a data
              source that is down gives HTTP 503, any other failure HTTP 500

    Example:
        run('score_location', score_location, 22.5530, 88.3540)
    """
    with _lock:
        try:
            result = jsonable(fn(*args, **kwargs))
        except UNAVAILABLE as err:
            raise HTTPException(503, f'{tool}: data source unavailable - '
                                     f'{type(err).__name__}: {redact(err)}')
        except Exception as err:
            raise HTTPException(500, f'{tool} failed - {type(err).__name__}: '
                                     f'{redact(err)}')
    result.setdefault('available', True)
    return result


@app.get('/', include_in_schema=False)
def home():
    """The bare address opens the interactive documentation."""
    return RedirectResponse('/docs')


@app.post('/tools/score_location', tags=[WRAPPED],
          summary='Score a location for site suitability using ML')
def tool_score_location(payload: LocationInput):
    """
    Site suitability of a location from the Phase I machine-learning model (Random
    Forest on the distance to the nearest road and hospital, OpenStreetMap in PostGIS):
    opportunity and risk scores from 0 to 10, a verdict and the SHAP contribution of
    each feature. Study area only.

    Args:
        payload (LocationInput): latitude and longitude in WGS84 decimal degrees

    Returns:
        dict: opportunity_score, risk_score, features (dist_road_m, dist_hospital_m,
              flood_risk), shap_explanation, verdict; outside the study area
              available = false and the reason

    Example:
        POST /tools/score_location  {"latitude": 22.5530, "longitude": 88.3540}
    """
    lat, lon = payload.latitude, payload.longitude
    if not in_study_area(lat, lon):
        return outside_study_area(lat, lon, 'Phase I site scorer',
                                  ('opportunity_score', 'risk_score', 'verdict'))
    return run('score_location', score_location, lat, lon)


@app.post('/tools/classify_imagery', tags=[WRAPPED],
          summary='Classify satellite imagery land cover at a location')
def tool_classify_imagery(payload: LocationInput):
    """
    Land cover within 500 m of a location from the Phase II segmentation model on the
    Sentinel-2 composite (Urban, Vegetation, Water, Bare Land, Agriculture), with NDVI,
    NDWI and the NDVI change since the baseline year. Study area only.

    Args:
        payload (LocationInput): latitude and longitude in WGS84 decimal degrees

    Returns:
        dict: land_cover, confidence_pct, class_distribution, ndvi, ndwi, ndvi_label,
              ndvi_change, change_flag, change_description, model, year; outside
              the study area available = false and the reason

    Example:
        POST /tools/classify_imagery  {"latitude": 22.5530, "longitude": 88.3540}
    """
    lat, lon = payload.latitude, payload.longitude
    if not in_study_area(lat, lon):
        return outside_study_area(lat, lon, 'Phase II Sentinel-2 composite',
                                  ('land_cover', 'confidence_pct', 'ndvi'))
    return run('classify_imagery', classify_imagery, lat, lon, radius_m=500)


@app.post('/tools/lidar_profile', tags=[WRAPPED],
          summary='Get LiDAR-derived building height, canopy, and slope metrics')
def tool_lidar_profile(payload: BboxInput):
    """
    Average building height, canopy cover, impervious surface and slope risk inside a
    bounding box, from the classified airborne LiDAR point cloud of Phase III (USGS
    3DEP, Dupont Circle tile, Washington DC).

    Args:
        payload (BboxInput): min_lat, min_lon, max_lat, max_lon in WGS84 degrees

    Returns:
        dict: avg_building_height_m, canopy_cover_pct, impervious_surface_pct,
              slope_risk_score (0-10), ground_relief_m, total_points_analyzed,
              lidar_coverage; a box without points gives available = false

    Example:
        POST /tools/lidar_profile  {"min_lat": 38.9085, "min_lon": -77.0450,
                                    "max_lat": 38.9110, "max_lon": -77.0420}
    """
    p = payload
    result = run('lidar_profile', analyze_bbox, p.min_lat, p.min_lon, p.max_lat, p.max_lon)
    if not result['lidar_coverage']:
        result.update(available=False, reason=result.pop('note', 'no LiDAR points'))
    return result


@app.post('/tools/streetview_scene', tags=[WRAPPED],
          summary='Get 360-degree street scene composition and walkability score')
def tool_streetview_scene(payload: LocationInput):
    """
    Street-level scene composition (road, sidewalk, building, vegetation, sky, person,
    car as % of pixels) and a walkability score from 0 to 10, from up to ten Mapillary
    photographs within about 165 m of a location, segmented with SegFormer (Phase III).
    Wherever Mapillary has imagery.

    Args:
        payload (LocationInput): latitude and longitude in WGS84 decimal degrees

    Returns:
        dict: scene_composition, walkability_score, images_analyzed, image_ids;
              available = false and the reason when there is no usable imagery

    Example:
        POST /tools/streetview_scene  {"latitude": 22.5530, "longitude": 88.3540}
    """
    result = run('streetview_scene', streetview_analyze, payload.latitude,
                 payload.longitude)
    if 'error' in result:
        result.update(available=False, reason=result.pop('error'))
    return result


@app.post('/tools/get_climate_risk', tags=[CONTEXT],
          summary='Get temperature anomaly and drought risk for a location')
def tool_get_climate_risk(payload: LocationInput):
    """
    Climate risk at a location: the temperature anomaly in degrees C and a drought
    index from 0 (normal or wetter) to 1 (extreme drought), and the flood-risk zone of
    the point (high, medium or low; study area only), from tools/climate.py; the source
    field names the data behind the values.

    Args:
        payload (LocationInput): latitude and longitude in WGS84 decimal degrees

    Returns:
        dict: temperature_anomaly_c, drought_index, flood_zone and source, with the
              classes, the precipitation and the period

    Example:
        POST /tools/get_climate_risk  {"latitude": 13.0827, "longitude": 80.2707}
    """
    return run('get_climate_risk', get_climate_risk, payload.latitude, payload.longitude)


@app.post('/tools/get_demographics', tags=[CONTEXT],
          summary='Get population density and healthcare access near a location')
def tool_get_demographics(payload: LocationInput):
    """
    Population density (people per km2) and access to healthcare (distance to the
    nearest hospital) around a location, from tools/demographics.py; the source field
    names the data behind the values.

    Args:
        payload (LocationInput): latitude and longitude in WGS84 decimal degrees

    Returns:
        dict: population_density_per_km2, nearest_hospital_km and source (with the
              version of Task 2.3 also the hospital name and the hospitals within 2 km)

    Example:
        POST /tools/get_demographics  {"latitude": 22.5530, "longitude": 88.3540}
    """
    return run('get_demographics', get_demographics, payload.latitude, payload.longitude)


@app.post('/tools/find_similar_sites', tags=[CONTEXT],
          summary='Find sites with a similar profile to a reference location')
def tool_find_similar_sites(payload: SimilarSitesInput):
    """
    Candidate sites with a profile similar to a reference location: the reference is
    scored with the Phase I model, and candidate sites of the Phase I grid within
    radius_km are ranked by the difference in opportunity score (tools/similar_sites.py).
    Study area only.

    Args:
        payload (SimilarSitesInput): latitude, longitude, radius_km (default 5),
            top_k (default 5)

    Returns:
        dict: reference (the score_location result) and similar_sites (ranked list);
              outside the study area available = false and the reason

    Example:
        POST /tools/find_similar_sites  {"latitude": 22.5530, "longitude": 88.3540}
    """
    p = payload
    return run('find_similar_sites', find_similar_sites, p.latitude, p.longitude,
               p.radius_km, p.top_k)


# -- The agent's latest result for the digital twin (Exercise 5) ----------------------
class AgentResult(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description='Latitude, WGS84')
    longitude: float = Field(..., ge=-180, le=180, description='Longitude, WGS84')
    question: str = Field('', description='The question the agent answered')
    answer: str = Field('', description='The start of its answer')
    tools: list[str] = Field([], description='The tools it called')
    sites: list[dict] = Field([], description='Similar sites to mark (rank, lat, lon)')
    source: str = Field('', description='Where the location came from')


LAST_AGENT_RESULT = {}
_result_ids = itertools.count(1)


@app.get('/agent/latest_result', tags=[AGENT],
         summary="Location of the agent's latest answer (polled by the twin)")
def get_latest_result():
    """
    The location and summary of the agent's latest answer; empty until the first one.

    Returns:
        dict: id (grows with every new result), latitude, longitude, question, answer,
              tools, sites, source and received

    Example:
        GET /agent/latest_result
    """
    return LAST_AGENT_RESULT


@app.post('/agent/latest_result', tags=[AGENT],
          summary="Store the location of the agent's latest answer")
def post_latest_result(payload: AgentResult):
    """
    Called by the agent after every answer, so that the twin can fly to the place.

    Args:
        payload (AgentResult): latitude, longitude and a summary of the answer

    Returns:
        dict: stored = true and the id of the result

    Example:
        POST /agent/latest_result  {"latitude": 22.5530, "longitude": 88.3540}
    """
    LAST_AGENT_RESULT.clear()
    LAST_AGENT_RESULT.update(payload.model_dump(), id=next(_result_ids),
                             received=datetime.now().isoformat(timespec='seconds'))
    return {'stored': True, 'id': LAST_AGENT_RESULT['id']}


# -- The same seven tools through the Model Context Protocol (/mcp) -------------------
mcp = MCPServer(name='GeoSense Agent MCP Server', version='1.0',
                instructions='Geospatial analysis tools of GeoSense Agent 2.0. '
                             + COVERAGE)
Lat = Annotated[float, Field(ge=-90, le=90, description='Latitude in WGS84 degrees')]
Lon = Annotated[float, Field(ge=-180, le=180, description='Longitude in WGS84 degrees')]
Km = Annotated[float, Field(gt=0, le=25, description='Search radius, km')]
TopK = Annotated[int, Field(ge=1, le=20, description='Number of sites to return')]
JSON = dict[str, Any]
READ_ONLY = ToolAnnotations(readOnlyHint=True)


def about(endpoint):
    """The first paragraph of an endpoint's docstring: the tool's MCP description."""
    return ' '.join(inspect.cleandoc(endpoint.__doc__).split('\n\n')[0].split())


def mcp_call(endpoint, model, **arguments):
    """
    Answer an MCP tools/call through the REST endpoint of the same tool, so that both
    interfaces validate and compute in exactly the same way.

    Args:
        endpoint (callable): the REST endpoint function, e.g. tool_score_location
        model (type): its Pydantic input model; **arguments are its fields

    Returns:
        dict: the endpoint's JSON result; a failure is raised as an MCP tool error
              whose text the client (an agent) can read

    Example:
        mcp_call(tool_score_location, LocationInput, latitude=22.553, longitude=88.354)
    """
    try:
        return endpoint(model(**arguments))
    except ValidationError as err:                   # e.g. bounding box corners reversed
        raise MCPToolError('; '.join(e['msg'] for e in err.errors()))
    except HTTPException as err:                     # 503 / 500 from run()
        raise MCPToolError(f'HTTP {err.status_code}: {err.detail}')


# Each MCP tool takes the fields of its REST route and answers through that route
@mcp.tool(name='score_location', description=about(tool_score_location),
          annotations=READ_ONLY)
def mcp_score_location(latitude: Lat, longitude: Lon) -> JSON:
    return mcp_call(tool_score_location, LocationInput,
                    latitude=latitude, longitude=longitude)


@mcp.tool(name='classify_imagery', description=about(tool_classify_imagery),
          annotations=READ_ONLY)
def mcp_classify_imagery(latitude: Lat, longitude: Lon) -> JSON:
    return mcp_call(tool_classify_imagery, LocationInput,
                    latitude=latitude, longitude=longitude)


@mcp.tool(name='lidar_profile', description=about(tool_lidar_profile),
          annotations=READ_ONLY)
def mcp_lidar_profile(min_lat: Lat, min_lon: Lon, max_lat: Lat, max_lon: Lon) -> JSON:
    return mcp_call(tool_lidar_profile, BboxInput, min_lat=min_lat, min_lon=min_lon,
                    max_lat=max_lat, max_lon=max_lon)


@mcp.tool(name='streetview_scene', description=about(tool_streetview_scene),
          annotations=READ_ONLY)
def mcp_streetview_scene(latitude: Lat, longitude: Lon) -> JSON:
    return mcp_call(tool_streetview_scene, LocationInput,
                    latitude=latitude, longitude=longitude)


@mcp.tool(name='get_climate_risk', description=about(tool_get_climate_risk),
          annotations=READ_ONLY)
def mcp_get_climate_risk(latitude: Lat, longitude: Lon) -> JSON:
    return mcp_call(tool_get_climate_risk, LocationInput,
                    latitude=latitude, longitude=longitude)


@mcp.tool(name='get_demographics', description=about(tool_get_demographics),
          annotations=READ_ONLY)
def mcp_get_demographics(latitude: Lat, longitude: Lon) -> JSON:
    return mcp_call(tool_get_demographics, LocationInput,
                    latitude=latitude, longitude=longitude)


@mcp.tool(name='find_similar_sites', description=about(tool_find_similar_sites),
          annotations=READ_ONLY)
def mcp_find_similar_sites(latitude: Lat, longitude: Lon, radius_km: Km = 5.0,
                           top_k: TopK = 5) -> JSON:
    return mcp_call(tool_find_similar_sites, SimilarSitesInput, latitude=latitude,
                    longitude=longitude, radius_km=radius_km, top_k=top_k)


mcp_app = mcp.streamable_http_app(streamable_http_path='/mcp', stateless_http=True,
                                  json_response=True)
app.router.routes.extend(mcp_app.routes)        # /mcp served by the same Uvicorn process


if __name__ == '__main__':
    import uvicorn
    tools = [r.path.split('/')[-1] for r in app.routes if r.path.startswith('/tools/')]
    print(f'GeoSense Agent MCP Server - {len(tools)} tools: {", ".join(tools)}')
    print(f'torch {torch.__version__}, device '
          f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print('REST /tools/<name>  |  docs http://localhost:8000/docs  |  '
          'MCP http://localhost:8000/mcp')
    print(f'CORS origins: {", ".join(CORS_ORIGINS)}')
    uvicorn.run(app, host='0.0.0.0', port=8000)
