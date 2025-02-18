import psycopg2
from psycopg2 import sql, Error
import pandas as pd
from db_config import DB_CONFIG
def get_db_connection():
    DB_USER     = DB_CONFIG["DB_USER"]
    DB_PASSWORD = DB_CONFIG["DB_PASSWORD"]
    DB_HOST     = DB_CONFIG["DB_HOST"]
    DB_PORT     = DB_CONFIG["DB_PORT"]
    DB_NAME     = DB_CONFIG["DB_NAME"]

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Error as e:
        print(f"Failed at connecting DB: {e}")
        return None
