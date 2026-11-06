# Demo question script — GeoSense Agent 2.0 (Part D.1)

Six questions for the recorded demo and for interviews, one per use case of the blueprint. Each line gives the tools the agent should select, the location it should resolve, and the answer time measured on the laptop in Phase 4 (the first call of a tool after a restart is slower: the LiDAR tile loads in about 14 s, Earth Engine answers in about 5 s before its cache is warm).

| # | Use case | Question to type | Tools expected | Location | Phase 4 timing |
|---|----------|------------------|----------------|----------|----------------|
| 1 | Clinic site selection (Phase 1) | Is Park Street in Kolkata a good site for a new clinic, and is it safe from flooding? | score_location, get_climate_risk | "Park Street, Kolkata" → 22.5552, 88.3501 | 9 s |
| 2 | Comparable sites (Phase 1 + PostGIS) | Which sites within 5 km of 22.5450, 88.3600 are similar to it, and are they any better? | score_location, find_similar_sites | stated coordinates | 9–10 s |
| 3 | Land cover and change (Phase 2) | What is the land cover at 22.5726, 88.3639 and has it changed since 2018? | classify_imagery | stated coordinates | 6 s |
| 4 | Buildings and canopy (Phase 3 LiDAR) | How tall are the buildings and how much tree cover is there around 38.9097, -77.0434? | lidar_profile | stated coordinates (Dupont Circle) | 16–20 s first call |
| 5 | Walkability (Phase 3 street imagery) | Is Park Street in Kolkata pleasant to walk along? | streetview_scene | "Park Street, Kolkata" | 25–30 s (10 images) |
| 6 | Population and climate (Phase 4 tools) | How many people live within 1 km of 22.5530, 88.3540, and was the past year unusually hot or dry there? | get_demographics, get_climate_risk | stated coordinates | 6–8 s |

Follow-up for the memory demonstration: after question 1, ask "What about its walkability?" — the location is carried over from the previous turn (Exercise 5 of Phase 4).

Order for the five-minute video (Task E.5.1): question 1 in the Streamlit app, the twin flying to Park Street, the follow-up, then question 2 with the five similar-site markers on the twin. Questions 3–6 are for interviews.
