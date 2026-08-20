# db_connection.py
# Purpose: single reusable entry point for connecting to geosense_db.
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
    return create_engine(conn_str)

def test_connection():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text('SELECT PostGIS_Version()'))
        print('Connected! PostGIS version:', result.fetchone()[0])

if __name__ == '__main__':
    test_connection()