# db_connection.py
# Purpose: single reusable entry point for connecting to geosense_db.
# Phase 5, Exercise 2: DB_SSLMODE (unset on the laptop; 'require' for a cloud database,
#   which accepts encrypted connections only) and pool_pre_ping, which tests a pooled
#   connection before reusing it - a cloud database closes idle connections, and the
#   server would otherwise hand the first request after a pause a dead connection.
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

def get_engine():
    """Creates and returns a database engine."""
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    name = os.getenv('DB_NAME', 'geosense_db')
    user = os.getenv('DB_USER', 'postgres')
    pwd  = os.getenv('DB_PASSWORD', 'postgres123')
    conn_str = f'postgresql://{user}:{pwd}@{host}:{port}/{name}'
    sslmode = os.getenv('DB_SSLMODE')              # 'require' in the cloud; unset here
    connect_args = {'sslmode': sslmode} if sslmode else {}
    return create_engine(conn_str, connect_args=connect_args, pool_pre_ping=True)

def test_connection():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text('SELECT PostGIS_Version()'))
        print('Connected! PostGIS version:', result.fetchone()[0])

if __name__ == '__main__':
    test_connection()