from db_connect import get_db_connection
import pandas as pd
# --------------------- 数据库查询函数 ---------------------
def query_movie_details(movie_id):
    """
    根据电影 ID 从 movies 表查询电影详细信息。
    """
    conn = get_db_connection()
    if conn is None:
        print("无法建立数据库连接。")
        return None
    try:
        query = f'SELECT * FROM movies WHERE "movieId" = {movie_id};'
        movie_details = pd.read_sql_query(query, conn)
        return movie_details
    except Exception as e:
        print("查询电影详情时出错：", e)
        return None
    finally:
        conn.close()

def query_user_ratings(user_id):
    """
    根据用户 ID 从 ratings 表查询该用户的评分记录。
    """
    conn = get_db_connection()
    if conn is None:
        print("无法建立数据库连接。")
        return None
    try:
        query = f'SELECT * FROM ratings WHERE "userId" = {user_id};'
        user_ratings = pd.read_sql_query(query, conn)
        return user_ratings
    except Exception as e:
        print("查询用户评分时出错：", e)
        return None
    finally:
        conn.close()