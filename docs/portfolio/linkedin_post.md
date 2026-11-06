# LinkedIn portfolio post — draft for the job-search launch (Task E.5.3)

Attach `docs/portfolio/demo.gif` with the **photo** button (a GIF added as a photo plays in the feed;
LinkedIn takes GIFs up to about 5 MB). Post text:

---

What if you could ask a map a question?

For my M.Tech in Geoinformatics I built GeoSense Agent 2.0 — a conversational geospatial
intelligence platform. You type a question such as "Is Park Street in Kolkata a good place for a
new clinic, and is it safe from flooding?", and an AI agent decides which analyses answer it, runs
them, and flies a 3D digital twin of the city to the site.

Behind the conversation:
• Deep learning on Sentinel-2 imagery (U-Net, with NASA–IBM's Prithvi foundation model compared)
• LiDAR building heights and tree canopy, SegFormer on street-level photographs
• A Random Forest site score explained with SHAP, PostGIS, Earth Engine climate and population data
• Seven analyses exposed as MCP tools to a LangGraph agent
• A CesiumJS digital twin with flood-risk zones, extruded buildings and a 2018–2025 time slider
• Docker on Google Cloud Run, with GitHub Actions testing and deploying every change

Live demo: https://geosense-agent.streamlit.app
3-minute walkthrough: https://youtu.be/Ovwtpb2bGgA
Code (MIT): https://github.com/anubhavGIS/geosense-agent-2

I wrote up the build in three articles — GeoAI concepts, MCP integration and the digital twin —
linked in the comment below. Feedback from people working in GIS, remote sensing and GeoAI is very
welcome.

#GeoAI #MCP #DigitalTwin #RemoteSensing #MachineLearning #GIS

---

First comment (posted right after the post):

> The three articles: Part 1 — https://github.com/anubhavGIS/geosense-agent-2/blob/main/docs/portfolio/blog/part1_geoai.md · Part 2 — https://github.com/anubhavGIS/geosense-agent-2/blob/main/docs/portfolio/blog/part2_mcp.md · Part 3 — https://github.com/anubhavGIS/geosense-agent-2/blob/main/docs/portfolio/blog/part3_digital_twin.md
