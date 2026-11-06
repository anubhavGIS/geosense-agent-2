# Demo script — presenting GeoSense Agent 2.0 live in five minutes (Task E.5.1)

The script for showing the live system in person — an interview, a viva, a screen share: the twin
on GitHub Pages, the Streamlit app on Streamlit Community Cloud and the backend on Google Cloud Run.
Questions come from [demo_questions.md](demo_questions.md). The published video
([YouTube, unlisted](https://youtu.be/Ovwtpb2bGgA)) tells the same story as a three-minute captioned walkthrough
assembled from screenshots of the live system.

| | |
|---|---|
| Twin | https://anubhavgis.github.io/geosense-agent-2/ |
| Demo app | https://geosense-agent.streamlit.app |
| Repository | https://github.com/anubhavGIS/geosense-agent-2 |

## Before presenting (10 minutes)

1. **Wake the backend.** Open https://geosense-backend-917306622260.us-central1.run.app/health and
   wait for the JSON (up to about 70 s when the service has scaled to zero).
2. **Wake the app.** Open the demo app; if it says the app has gone to sleep, press the button that
   wakes it and wait for the question box.
3. **Warm the tools.** Ask question 1 once and wait for the answer (the first call of each tool loads
   its model), then press **New conversation**.
4. **A clean screen.** A Chrome window with the three tabs above (twin, app, repository) and nothing
   else; bookmarks bar hidden (Ctrl+Shift+B); Windows notifications off (Do not disturb).

## The five minutes

| Time | Show | Say (about this much) |
|---|---|---|
| 0:00–0:30 | **Intro** — the repository page, top of the README | "Planners, hospitals and disaster teams ask location questions every day — where to put a clinic, which areas flood, how green a district is. Today that takes GIS experts and many tools that do not talk to each other. GeoSense Agent 2.0 lets you simply ask." |
| 0:30–1:30 | **The twin** (tab 1) — drag to rotate, scroll to zoom over Kolkata (flood-risk zones and the 2025 imagery are on when the page opens); press **Loss hotspot**, then the year buttons **2018** and **2025**; untick **Sentinel-2 images**, tick **NDVI 2023**; press **Dupont Circle** and click one extruded building | "This is the 3D digital twin, built with CesiumJS on real terrain. Red, orange and yellow are the high, medium and low flood-risk zones, from the height above the nearest drainage. The year buttons compare Sentinel-2 imagery from 2018 and 2025 — here vegetation has turned into buildings. NDVI shows how green the city is. In Washington, D.C. the buildings are extruded from a LiDAR point cloud; the info box gives each one's height." |
| 1:30–3:30 | **The question** (tab 2) — question 1 is in the box; press **Ask GeoSense Agent**; while it runs, point at the spinner; then the answer, the **tools** table and the caption; then the follow-up *What about its walkability?* | "I ask in plain English: is Park Street a good place for a new clinic, and is it safe from flooding? The agent, a LangGraph graph with a Gemini model, reads the question and chooses two of its seven MCP tools: the machine-learning site score and the climate-and-flood tool, which run on Google Cloud Run. The answer cites its evidence: a good site — close to roads and a hospital — but inside a high flood-risk zone. The table shows each tool call, its status and its time. A follow-up does not repeat the place: the agent remembers it and calls the street-level tool, which segments Mapillary photographs with SegFormer." |
| 3:30–4:30 | **The result on the twin** — scroll down to the twin under the answer: it has flown to Park Street (yellow marker); ask question 2 and watch the camera move out to the five cyan markers | "Every answer is published to the twin, which flies to the site. Now: which sites within 5 km are similar, and are they any better? The agent scores the point and searches the 1,125 candidate sites in PostGIS; the twin shows the five most similar ones and their ranks." |
| 4:30–5:00 | **GitHub** (tab 3) — the badges, the live links, the Actions tab with green runs | "Everything is open source: the code of all five phases, the tests and the pipeline that tests every push and deploys it to Cloud Run. The link is in the description — github.com/anubhavGIS/geosense-agent-2." |

Question 1 (already in the app's box): *Is Park Street in Kolkata (22.5530, 88.3540) a good place
for a new clinic, and is it safe from flooding?* — Follow-up: *What about its walkability?* —
Question 2: *Which sites within 5 km of 22.5450, 88.3600 are similar to it, and are they any better?*

## The video and the GIF

- `GeoSense_Agent_2_walkthrough` — 1920 × 1080, 30 fps, 3 min 9 s, no sound: fourteen captioned
  scenes (the problem, the architecture, the live answer, the twin, similar sites, the LiDAR
  buildings, the 2018/2025 imagery, the four analyses, the pipeline, the links), each a still
  image encoded with ffmpeg and joined; uploaded to YouTube as **unlisted**: https://youtu.be/Ovwtpb2bGgA.
- `demo.gif` — five of those scenes (the answer, the twin, similar sites, buildings, the years),
  2 s each, 960 × 540, one 256-colour palette, about 0.9 MB, for the LinkedIn post.
