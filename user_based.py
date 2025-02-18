#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import psycopg2
from psycopg2 import Error
from db_connect import get_db_connection
from db_query import query_movie_details, query_user_ratings

MODEL_FILE = "model/user_based_model.pkl"

# --------------------- 1. 从数据库加载数据 ---------------------
def load_data_from_db():
    conn = get_db_connection()
    print(conn)
    if conn is None:
        print("Failed to connect to the database.")
        return None, None

    try:
        ratings_query = 'SELECT "userId", "movieId", "rating", "timestamp" FROM ratings'
        movies_query = 'SELECT "movieId", "title", "genres" FROM movies'
        ratings = pd.read_sql_query(ratings_query, conn)
        movies = pd.read_sql_query(movies_query, conn)
        print("Successfully loaded data from database.")
        return ratings, movies
    except Error as e:
        print(f"Failed: {e}")
        return None, None
    finally:
        conn.close()

# --------------------- 2. 用户协同过滤类 ---------------------
class UserBasedCF:
    def __init__(self, n_sim_user=20, n_rec_movie=10, pivot=0.75):
        self.n_sim_user = n_sim_user      # 用于计算相似用户的数量
        self.n_rec_movie = n_rec_movie    # 每个用户推荐的电影数量
        self.pivot = pivot                # 数据集划分比例，约有 75% 数据用作训练
        self.trainSet = {}                # 训练集：{user: {movie: rating, ...}, ...}
        self.testSet = {}                 # 测试集：  {user: {movie: rating, ...}, ...}
        self.user_list = None             # 训练集中所有用户的列表（后续用于索引）
        self.sim_matrix = None            # 用户相似度矩阵

    def get_dataset(self):
        """
        从数据库中加载评分数据，然后按照 pivot 值随机划分为训练集和测试集。
        """
        ratings, _ = load_data_from_db()
        if ratings is None:
            print("Failed to load ratings from database.")
            return

        for _, row in ratings.iterrows():
            user = str(row['userId'])
            movie = str(row['movieId'])
            rating = float(row['rating'])
            if np.random.random() < self.pivot:
                self.trainSet.setdefault(user, {})[movie] = rating
            else:
                self.testSet.setdefault(user, {})[movie] = rating

        print("Split training and test dataset successfully!")
        print(f"TrainSet size: {sum(len(movies) for movies in self.trainSet.values())}")
        print(f"TestSet size: {sum(len(movies) for movies in self.testSet.values())}")

    def calc_user_sim_sparse(self):
        """
        将训练集转换为用户-电影稀疏矩阵，并用向量化方式计算余弦相似度，
        得到用户之间的相似度矩阵。
        """
        self.user_list = list(self.trainSet.keys())
        movie_set = set()
        for user in self.user_list:
            movie_set.update(self.trainSet[user].keys())
        movie_list = list(movie_set)
        
        # 构造索引映射
        user_to_idx = {u: i for i, u in enumerate(self.user_list)}
        movie_to_idx = {m: j for j, m in enumerate(movie_list)}
        
        data, rows, cols = [], [], []
        for user in self.user_list:
            for movie, rating in self.trainSet[user].items():
                rows.append(user_to_idx[user])
                cols.append(movie_to_idx[movie])
                data.append(rating)
        sparse_mat = csr_matrix((data, (rows, cols)), shape=(len(self.user_list), len(movie_list)))
        print(f"Sparse matrix shape: {sparse_mat.shape}")
        
        self.sim_matrix = cosine_similarity(sparse_mat)
        print("Calculated user similarity matrix using sparse matrix.")

    def recommend_sparse(self, target_user):
        """
        基于预先计算好的相似度矩阵，为目标用户推荐电影。
        """
        if target_user not in self.trainSet:
            return []
        user_to_idx = {u: i for i, u in enumerate(self.user_list)}
        target_idx = user_to_idx[target_user]
        sim_vector = self.sim_matrix[target_idx]
        
        # 取出与目标用户相似度最高的 n_sim_user 个用户（排除自己）
        similar_idxs = np.argsort(-sim_vector)
        similar_user_idxs = [i for i in similar_idxs if self.user_list[i] != target_user][:self.n_sim_user]
        
        rank = {}
        watched_movies = set(self.trainSet[target_user].keys())
        for idx in similar_user_idxs:
            sim = self.sim_matrix[target_idx, idx]
            other_user = self.user_list[idx]
            for movie, rating in self.trainSet[other_user].items():
                if movie in watched_movies:
                    continue
                rank[movie] = rank.get(movie, 0) + sim  # 累加相似度作为得分
        recommended = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:self.n_rec_movie]
        return recommended

    def evaluate_model(self, k=None):
        """
        利用离线测试集评价推荐系统的性能。
        
        评价指标包括：
          - Precision@K：在前 K 个推荐中相关项目的比例。
          - Recall@K：测试集中相关项目被推荐的比例。
          - F1@K：Precision 与 Recall 的调和平均值。
          - Coverage：所有用户推荐中不同电影数占训练集电影总数的比例。
          
        k 若未设置，则默认为 n_rec_movie。
        """
        if k is None:
            k = self.n_rec_movie

        precisions = []
        recalls = []
        f1_scores = []
        n_users_evaluated = 0

        # 针对测试集中的用户进行评价（仅处理同时存在于训练集的用户）
        for user in self.testSet:
            if user not in self.trainSet:
                continue
            ground_truth = set(self.testSet[user].keys())
            recommendations = self.recommend_sparse(user)
            rec_items = set([movie for movie, score in recommendations])
            if not rec_items:
                continue
            hit_count = len(rec_items.intersection(ground_truth))
            precision = hit_count / k
            recall = hit_count / len(ground_truth)
            precisions.append(precision)
            recalls.append(recall)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)
            n_users_evaluated += 1

        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0

        # 计算覆盖率：所有用户中推荐过的不同电影数占训练集中电影总数的比例
        rec_items_all = set()
        for user in self.trainSet:
            recs = self.recommend_sparse(user)
            for movie, score in recs:
                rec_items_all.add(movie)
        all_train_movies = set()
        for user in self.trainSet:
            all_train_movies.update(self.trainSet[user].keys())
        coverage = len(rec_items_all) / len(all_train_movies) if all_train_movies else 0

        metrics = {
            "Precision@{}".format(k): avg_precision,
            "Recall@{}".format(k): avg_recall,
            "F1@{}".format(k): avg_f1,
            "Coverage": coverage
        }
        print(f"Evaluated on {n_users_evaluated} users from test set.")
        return metrics

    def save_model(self, filepath=MODEL_FILE):
        """
        保存模型的关键状态（包括训练集、测试集、用户列表、相似度矩阵及基本参数）
        到指定文件，以便后续加载重用。
        """
        state = {
            "n_sim_user": self.n_sim_user,
            "n_rec_movie": self.n_rec_movie,
            "pivot": self.pivot,
            "trainSet": self.trainSet,
            "testSet": self.testSet,
            "user_list": self.user_list,
            "sim_matrix": self.sim_matrix
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath=MODEL_FILE):
        """
        从指定文件加载之前保存的模型状态。
        """
        if not os.path.exists(filepath):
            print(f"模型文件 {filepath} 不存在")
            return False
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.n_sim_user = state.get("n_sim_user")
        self.n_rec_movie = state.get("n_rec_movie")
        self.pivot = state.get("pivot")
        self.trainSet = state.get("trainSet")
        self.testSet = state.get("testSet")
        self.user_list = state.get("user_list")
        self.sim_matrix = state.get("sim_matrix")
        print(f"从 {filepath} 加载模型成功")
        return True

# --------------------- 3. 使用示例 ---------------------
if __name__ == '__main__':
    userCF = UserBasedCF(n_sim_user=20, n_rec_movie=10, pivot=0.75)
    
    # 尝试加载模型，如果不存在则训练并保存模型
    if os.path.exists(MODEL_FILE):
        loaded = userCF.load_model()
        if not loaded:
            print("加载模型失败，将重新训练")
            userCF.get_dataset()
            userCF.calc_user_sim_sparse()
            userCF.save_model()
    else:
        userCF.get_dataset()
        userCF.calc_user_sim_sparse()
        userCF.save_model()
    
    # # 评价推荐系统性能（离线评价）
    # metrics = userCF.evaluate_model()
    # print("Evaluation Metrics:")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value:.4f}")
    
    # 测试推荐：选取训练集中第一个用户进行推荐
    # 输入用户ID进行查询
    if userCF.user_list:
        target_user = input("请输入要推荐的用户ID（userId）：").strip()
        target_user = str(float(target_user))
        recs = userCF.recommend_sparse(target_user)
        print(f"\nRecommendations for user {target_user}:")
        for movie, score in recs:
            # 查询电影详细信息
            movie_details = query_movie_details(movie)
            if movie_details is not None and not movie_details.empty:
                title = movie_details.iloc[0]['title']
                genres = movie_details.iloc[0]['genres']
            else:
                title, genres = "N/A", "N/A"
            print(f"MovieId: {movie} | Title: {title} | Genres: {genres} | CF Score: {score:.4f}")
        
        # 输出目标用户在数据库中的所有评分记录
        print(f"\n用户 {target_user} 的评分记录：")
        
        user_ratings = query_user_ratings(target_user)
        if user_ratings is not None and not user_ratings.empty:
            # 按rating降序排列，取最高的5条记录
            top5 = user_ratings.sort_values(by="rating", ascending=False).head(5)
            print(f"\n用户 {target_user} 评分最高的5部电影：")
            for index, row in top5.iterrows():
                movie_id = row["movieId"]
                # 根据movie_id查询电影详细信息
                movie_details = query_movie_details(movie_id)
                if movie_details is not None and not movie_details.empty:
                    title = movie_details.iloc[0]['title']
                    genres = movie_details.iloc[0]['genres']
                else:
                    title, genres = "N/A", "N/A"
                print(f"MovieId: {movie_id} | Title: {title} | Genres: {genres} | Rating: {row['rating']}")
        else:
            print("该用户在数据库中暂无评分记录。")
    else:
        print("训练集为空，无法给出推荐。")
