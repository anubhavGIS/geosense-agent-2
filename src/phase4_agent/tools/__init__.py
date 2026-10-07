# tools/ — one module per MCP tool whose logic is more than a call to an earlier phase
#   common.py         study-area gate, JSON conversion, error type shared by the tools
#   climate.py        get_climate_risk    (temperature anomaly and drought risk)
#   demographics.py   get_demographics    (population density and hospital access)
#   similar_sites.py  find_similar_sites  (candidate sites with a similar profile)
# The four tools that wrap a Phase I-III module directly live in mcp_server.py.
