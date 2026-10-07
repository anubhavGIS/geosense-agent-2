# mcp_client_demo.py
# Purpose: use the server the way an MCP client does (an agent framework, Claude Desktop,
#          an IDE): connect to /mcp, discover the tools with tools/list - name,
#          description, input schema - and call them with tools/call
# Run from the project root while mcp_server.py runs in another window:
#   python src/phase4_agent/mcp_client_demo.py
import asyncio
import json
import os
import textwrap

from dotenv import load_dotenv
from mcp import Client

load_dotenv()
URL = os.getenv('MCP_SERVER_URL', 'http://localhost:8000') + '/mcp'
CALLS = [('score_location', {'latitude': 22.5530, 'longitude': 88.3540}),
         ('get_climate_risk', {'latitude': 13.0827, 'longitude': 80.2707}),
         ('lidar_profile', {'min_lat': 38.92, 'min_lon': -77.045,      # corners
                            'max_lat': 38.911, 'max_lon': -77.042})]     # reversed


async def main():
    """
    List the tools of the server and call three of them through the MCP protocol.

    Args:
        none - the server address comes from MCP_SERVER_URL in .env

    Returns:
        None: prints the server, each tool with its parameters and description, and
              the result (or the error) of each call

    Example:
        asyncio.run(main())
    """
    async with Client(URL) as client:
        info = client.server_info
        print(f'Connected to {info.name} {info.version} at {URL} '
              f'(MCP protocol {client.protocol_version})')
        tools = (await client.list_tools()).tools
        print(f'\ntools/list -> {len(tools)} tools')
        for tool in tools:
            schema = tool.input_schema
            params = ', '.join(name + ('' if name in schema.get('required', []) else '?')
                               for name in schema['properties'])
            print(f'  {tool.name}({params})')
            print(textwrap.fill(tool.description, 88, initial_indent=' ' * 6,
                                subsequent_indent=' ' * 6, max_lines=2))
        for name, arguments in CALLS:
            print(f'\ntools/call -> {name} {json.dumps(arguments)}')
            result = await client.call_tool(name, arguments)
            if result.is_error:
                print('  is_error = True: ' + result.content[0].text)
                continue
            for key, value in result.structured_content.items():
                print(f'  {key:<26}{json.dumps(value)[:60]}')


if __name__ == '__main__':
    asyncio.run(main())
