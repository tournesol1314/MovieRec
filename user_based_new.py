#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import psycopg2
from psycopg2 import Error
from db_connect import get_db_connection
from db_query import query_movie_details, query_user_ratings

# 基础模型文件名
MODEL_BASE = "model/cf_model"
SVD_MODEL_BASE = "model/svd_model"

# --------------------- 1. 从数据库加载数据 ---------------------
def load_data_from_db():
    conn = get_db_connection()
    print(conn)
    if conn is None:
        print("Failed to connect to the database.")
        return None, None

    try:
        ratings_query = 'SELECT "userid", "movieid", "rating", "timestamp" FROM ratings'
        movies_query = 'SELECT "movieid", "title", "genres" FROM movies'
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
class EnhancedCF:
    def __init__(self, n_sim_user=20, n_rec_movie=10, pivot=0.75, n_factors=50, sim_method="Cosine"):
        self.n_sim_user = n_sim_user      # 用于计算相似用户的数量
        self.n_rec_movie = n_rec_movie    # 每个用户推荐的电影数量
        self.pivot = pivot                # 数据集划分比例，约有 75% 数据用作训练
        self.n_factors = n_factors        # SVD降维的因子数量
        self.sim_method = sim_method      # 相似度计算方法: "Cosine", "Pearson", "Manhattan"
        self.trainSet = {}                # 训练集：{user: {movie: rating, ...}, ...}
        self.testSet = {}                 # 测试集：  {user: {movie: rating, ...}, ...}
        self.user_list = None             # 训练集中所有用户的列表（后续用于索引）
        self.movie_list = None            # 训练集中所有电影的列表
        self.sim_matrix = None            # 用户相似度矩阵
        self.user_features = None         # SVD分解后的用户特征矩阵
        self.movie_features = None        # SVD分解后的电影特征矩阵
        self.sigma = None                 # SVD分解的奇异值
        self.use_svd = False              # 是否使用SVD模式
        self.mean_ratings = None          # 用户平均评分（用于SVD填充缺失值）
        self.user_to_idx = None           # 用户ID到矩阵索引的映射
        self.movie_to_idx = None          # 电影ID到矩阵索引的映射
        self.user_ratings_matrix = None   # 用户-电影评分矩阵（密集格式，用于Pearson相似度计算）

    def get_dataset(self):
        """
        从数据库中加载评分数据，然后按照 pivot 值随机划分为训练集和测试集。
        """
        ratings, _ = load_data_from_db()
        if ratings is None:
            print("Failed to load ratings from database.")
            return

        count = 0
        for _, row in ratings.iterrows():
            user = str(row['userid'])
            movie = str(row['movieid'])
            rating = float(row['rating'])
            if np.random.random() < self.pivot:
                self.trainSet.setdefault(user, {})[movie] = rating
                count += 1
            else:
                self.testSet.setdefault(user, {})[movie] = rating
            if count >= 1000000:  # 限制处理的数据量
                break

        print("Split training and test dataset successfully!")
        print(f"TrainSet size: {sum(len(movies) for movies in self.trainSet.values())}")
        print(f"TestSet size: {sum(len(movies) for movies in self.testSet.values())}")

    def _build_matrix(self):
        """
        构建用户-电影评分矩阵（稀疏表示）
        """
        self.user_list = list(self.trainSet.keys())
        movie_set = set()
        for user in self.user_list:
            movie_set.update(self.trainSet[user].keys())
        self.movie_list = list(movie_set)
        
        # 构造索引映射
        self.user_to_idx = {u: i for i, u in enumerate(self.user_list)}
        self.movie_to_idx = {m: j for j, m in enumerate(self.movie_list)}
        
        data, rows, cols = [], [], []
        for user in self.user_list:
            for movie, rating in self.trainSet[user].items():
                rows.append(self.user_to_idx[user])
                cols.append(self.movie_to_idx[movie])
                data.append(rating)
        sparse_mat = csr_matrix((data, (rows, cols)), shape=(len(self.user_list), len(self.movie_list)))
        print(f"Sparse matrix shape: {sparse_mat.shape}")
        
        return sparse_mat

    def _compute_similarity(self, matrix, method="Cosine"):
        """
        根据选择的方法计算相似度矩阵
        支持的方法: Cosine, Pearson, Manhattan
        """
        n_users = matrix.shape[0]
        sim_matrix = np.zeros((n_users, n_users))
        
        if method == "Cosine":
            # 余弦相似度 - 使用sklearn内置函数
            return cosine_similarity(matrix)
            
        elif method == "Manhattan":
            # 曼哈顿距离 - 转换为相似度（1/(1+距离)）
            dist_matrix = manhattan_distances(matrix)
            # 避免除零错误
            dist_matrix = np.where(dist_matrix == 0, 1e-10, dist_matrix)
            return 1.0 / (1.0 + dist_matrix)
            
        elif method in ["Pearson"]:
            # 对于Pearson，需要逐对计算
            # 转换为密集矩阵
            dense_matrix = matrix.toarray() if hasattr(matrix, 'toarray') else matrix
            
            for i in range(n_users):
                for j in range(i, n_users):
                    if i == j:
                        sim_matrix[i, j] = 1.0
                        continue
                    
                    # 获取两个用户共同评分的电影索引
                    u1_ratings = dense_matrix[i]
                    u2_ratings = dense_matrix[j]
                    mask = np.logical_and(u1_ratings > 0, u2_ratings > 0)
                    
                    # 如果没有共同评分，相似度为0
                    if np.sum(mask) < 2:  # 至少需要2个共同评分项
                        sim_matrix[i, j] = 0.0
                        sim_matrix[j, i] = 0.0
                        continue
                    
                    u1_common = u1_ratings[mask]
                    u2_common = u2_ratings[mask]
                    
                    if method == "Pearson":
                        # 皮尔逊相关系数
                        try:
                            corr, _ = pearsonr(u1_common, u2_common)
                            if np.isnan(corr):
                                corr = 0.0
                        except:
                            corr = 0.0
                    
                    # 将相关系数转换为[0,1]范围的相似度
                    # 相关系数范围是[-1,1]，转换为[0,1]
                    sim = (corr + 1) / 2.0
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
            
            return sim_matrix
        else:
            print(f"未知的相似度计算方法: {method}，使用默认的余弦相似度")
            return cosine_similarity(matrix)

    def calc_user_sim_sparse(self):
        """
        使用传统方法计算用户相似度矩阵
        """
        sparse_mat = self._build_matrix()
        self.sim_matrix = self._compute_similarity(sparse_mat, self.sim_method)
        print(f"计算用户相似度矩阵使用 {self.sim_method} 方法")
        self.use_svd = False

    def calc_user_sim_svd(self):
        """
        使用SVD分解用户-电影评分矩阵，并基于降维后的用户特征计算相似度
        """
        sparse_mat = self._build_matrix()
        
        # 计算每个用户的平均评分
        self.mean_ratings = np.zeros(sparse_mat.shape[0])
        for i in range(sparse_mat.shape[0]):
            row = sparse_mat.getrow(i).toarray().flatten()
            non_zero_indices = np.nonzero(row)[0]
            if len(non_zero_indices) > 0:
                self.mean_ratings[i] = np.mean(row[non_zero_indices])
            else:
                # 如果用户没有评分，使用全局平均分
                self.mean_ratings[i] = 3.0  # 假设评分范围是1-5
        
        # 转换为密集矩阵并填充缺失值
        ratings_dense = sparse_mat.toarray()
        ratings_centered = ratings_dense.copy()
        
        # 对缺失值填充用户平均评分，并对已有评分进行中心化
        for i in range(ratings_dense.shape[0]):
            zero_indices = ratings_dense[i] == 0
            ratings_centered[i, zero_indices] = 0  # 缺失值不参与SVD计算
            non_zero_indices = ~zero_indices
            ratings_centered[i, non_zero_indices] -= self.mean_ratings[i]  # 中心化
        
        # 执行SVD分解
        k = min(self.n_factors, min(ratings_centered.shape) - 1)  # 确保k不超过矩阵的维度
        U, sigma, Vt = svds(ratings_centered, k=k)
        
        # 注意：svds返回的奇异值是按升序排列的，需要反转
        U = np.flip(U, axis=1)
        sigma = np.flip(sigma)
        Vt = np.flip(Vt, axis=0)
        
        # 保存SVD分解结果
        self.user_features = U
        self.sigma = sigma
        self.movie_features = Vt.T
        
        # 计算用户相似度（基于降维后的用户特征）
        weighted_features = U.dot(np.diag(sigma))
        
        # 使用选择的相似度计算方法
        self.sim_matrix = self._compute_similarity(weighted_features, self.sim_method)
        
        print(f"计算用户相似度矩阵使用SVD ({k}个因子) 和 {self.sim_method} 相似度方法")
        self.use_svd = True

    def recommend_sparse(self, target_user):
        """
        基于传统协同过滤为目标用户推荐电影。
        """
        if target_user not in self.trainSet or target_user not in self.user_to_idx:
            return []
        
        target_idx = self.user_to_idx[target_user]
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

    def recommend_svd(self, target_user):
        """
        基于SVD矩阵分解为目标用户推荐电影。
        """
        if target_user not in self.trainSet or target_user not in self.user_to_idx:
            return []
        
        target_idx = self.user_to_idx[target_user]
        
        # 获取用户在潜在空间的特征向量
        user_vec = self.user_features[target_idx]
        
        # 预测所有电影的评分
        pred_ratings = self.mean_ratings[target_idx] + np.dot(user_vec * self.sigma, self.movie_features.T)
        
        # 过滤掉已观看的电影
        watched_movies = set(self.trainSet[target_user].keys())
        watched_indices = [self.movie_to_idx[m] for m in watched_movies if m in self.movie_to_idx]
        
        # 构建推荐列表
        candidate_indices = [i for i in range(len(self.movie_list)) if i not in watched_indices]
        candidate_ratings = [(self.movie_list[i], pred_ratings[i]) for i in candidate_indices]
        
        # 返回预测评分最高的电影
        recommended = sorted(candidate_ratings, key=lambda x: x[1], reverse=True)[:self.n_rec_movie]
        return recommended

    def recommend(self, target_user):
        """
        根据模型类型选择相应的推荐方法
        """
        if self.use_svd:
            return self.recommend_svd(target_user)
        else:
            return self.recommend_sparse(target_user)

    def evaluate_model(self, k=None):
        """
        评估推荐系统性能
        """
        if k is None:
            k = self.n_rec_movie

        precisions = []
        recalls = []
        f1_scores = []
        n_users_evaluated = 0

        # 针对测试集中的用户进行评价
        for user in self.testSet:
            if user not in self.trainSet:
                continue
            ground_truth = set(self.testSet[user].keys())
            if not ground_truth:
                continue
                
            recommendations = self.recommend(user)
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

        # 计算覆盖率
        rec_items_all = set()
        sample_users = random.sample(list(self.trainSet.keys()), min(100, len(self.trainSet)))
        for user in sample_users:
            recs = self.recommend(user)
            for movie, score in recs:
                rec_items_all.add(movie)
        
        all_train_movies = set(self.movie_list)
        coverage = len(rec_items_all) / len(all_train_movies) if all_train_movies else 0

        metrics = {
            "Precision@{}".format(k): avg_precision,
            "Recall@{}".format(k): avg_recall,
            "F1@{}".format(k): avg_f1,
            "Coverage": coverage,
            "Model Type": "SVD" if self.use_svd else "Traditional CF"
        }
        
        print(f"Evaluated on {n_users_evaluated} users from test set.")
        return metrics

    def save_model(self, filepath=None):
        """
        保存模型状态
        """
        if filepath is None:
            # 根据模型类型和相似度方法构建文件名
            model_type = "svd" if self.use_svd else "cf"
            method = self.sim_method.lower()
            filepath = f"model/{model_type}_{method}_model.pkl"
            
        state = {
            "n_sim_user": self.n_sim_user,
            "n_rec_movie": self.n_rec_movie,
            "pivot": self.pivot,
            "n_factors": self.n_factors,
            "sim_method": self.sim_method,
            "trainSet": self.trainSet,
            "testSet": self.testSet,
            "user_list": self.user_list,
            "movie_list": self.movie_list,
            "sim_matrix": self.sim_matrix,
            "use_svd": self.use_svd,
            "user_to_idx": self.user_to_idx,
            "movie_to_idx": self.movie_to_idx
        }
        
        if self.use_svd:
            state.update({
                "user_features": self.user_features,
                "movie_features": self.movie_features,
                "sigma": self.sigma,
                "mean_ratings": self.mean_ratings
            })
            
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath=None, model_type=None, sim_method=None):
        """
        加载模型状态
        
        参数:
            filepath: 模型文件路径，如果为None则根据model_type和sim_method自动查找
            model_type: 模型类型 ("svd" 或 "cf")
            sim_method: 相似度方法 ("cosine", "pearson", "manhattan")
        """
        if filepath is None:
            if model_type is not None and sim_method is not None:
                # 根据指定的模型类型和相似度方法构建文件名
                filepath = f"model/{model_type}_{sim_method.lower()}_model.pkl"
            else:
                # 寻找任何可用的模型文件
                model_files = []
                for mt in ["svd", "cf"]:
                    for sm in ["cosine", "pearson", "manhattan"]:
                        file_path = f"model/{mt}_{sm}_model.pkl"
                        if os.path.exists(file_path):
                            model_files.append(file_path)
                
                if model_files:
                    # 默认加载第一个找到的模型文件
                    filepath = model_files[0]
                    print(f"未指定模型文件，将加载: {filepath}")
                else:
                    print("没有找到任何模型文件")
                    return False
                
        if not os.path.exists(filepath):
            print(f"模型文件 {filepath} 不存在")
            return False
            
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            
        self.n_sim_user = state.get("n_sim_user")
        self.n_rec_movie = state.get("n_rec_movie")
        self.pivot = state.get("pivot")
        self.n_factors = state.get("n_factors")
        self.trainSet = state.get("trainSet")
        self.testSet = state.get("testSet")
        self.user_list = state.get("user_list")
        self.movie_list = state.get("movie_list")
        self.sim_matrix = state.get("sim_matrix")
        self.use_svd = state.get("use_svd", False)
        self.user_to_idx = state.get("user_to_idx")
        self.movie_to_idx = state.get("movie_to_idx")
        self.sim_method = state.get("sim_method", "Cosine")  # 默认为余弦相似度
        
        if self.use_svd:
            self.user_features = state.get("user_features")
            self.movie_features = state.get("movie_features")
            self.sigma = state.get("sigma")
            self.mean_ratings = state.get("mean_ratings")
            
        print(f"从 {filepath} 加载模型成功")
        print(f"模型类型: {'SVD' if self.use_svd else '传统协同过滤'}")
        print(f"相似度方法: {self.sim_method}")
        return True

# --------------------- 3. 使用示例 ---------------------
if __name__ == '__main__':
    # 创建增强版协同过滤模型实例，使用默认参数
    cf = EnhancedCF(n_sim_user=30, n_rec_movie=10, pivot=0.8, n_factors=35, sim_method="Cosine")
    
    # 尝试加载模型，如果不存在则训练
    if not cf.load_model():
        print("模型加载失败，将重新训练")
        cf.get_dataset()
        
        # 提示用户选择模型类型和相似度方法
        print("\n选择模型参数:")
        
        # 选择模型类型
        model_type = input("模型类型 (1=传统协同过滤, 2=SVD分解) [默认=2]: ").strip()
        use_svd = True if not model_type or model_type == "2" else False
        
        # 选择相似度计算方法
        print("\n相似度计算方法:")
        print("1: Cosine (余弦相似度) - 适用于大多数场景，计算速度快")
        print("2: Pearson (皮尔逊相关系数) - 考虑用户评分尺度的差异")
        print("3: Manhattan (曼哈顿距离) - 对异常值不敏感")
        
        sim_choice = input("选择相似度计算方法 [默认=1]: ").strip()
        sim_methods = {
            "1": "Cosine",
            "2": "Pearson", 
            "3": "Manhattan"
        }
        cf.sim_method = sim_methods.get(sim_choice, "Cosine")
        
        if use_svd:
            print(f"使用SVD分解计算用户相似度，使用{cf.sim_method}方法...")
            cf.calc_user_sim_svd()
        else:
            print(f"使用传统协同过滤计算用户相似度，使用{cf.sim_method}方法...")
            cf.calc_user_sim_sparse()
            
        cf.save_model()
    
    # 评估模型性能
    metrics = cf.evaluate_model()
    print("\n模型评估指标:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # 用户推荐
    print("\n电影推荐系统")
    print(f"当前使用模型: {'基于SVD的矩阵分解' if cf.use_svd else '传统协同过滤'}")
    print(f"相似度计算方法: {cf.sim_method}")
    
    while True:
        target_user = input("\n请输入要推荐的用户ID (q退出): ").strip()
        if target_user.lower() == 'q':
            break
            
        try:
            target_user = str(float(target_user))
            
            # 获取推荐结果
            recs = cf.recommend(target_user)
            
            if not recs:
                print(f"无法为用户 {target_user} 提供推荐，可能该用户不在训练集中或数据不足。")
                continue
                
            print(f"\n为用户 {target_user} 的电影推荐:")
            for movie, score in recs:
                # 查询电影详细信息
                movie_details = query_movie_details(movie)
                if movie_details is not None and not movie_details.empty:
                    title = movie_details.iloc[0]['title']
                    genres = movie_details.iloc[0]['genres']
                else:
                    title, genres = "未知", "未知"
                print(f"电影ID: {movie} | 标题: {title} | 类型: {genres} | 推荐分数: {score:.4f}")
            
            # 显示用户评分最高的电影
            print(f"\n用户 {target_user} 评分最高的电影:")
            user_ratings = query_user_ratings(target_user)
            if user_ratings is not None and not user_ratings.empty:
                top5 = user_ratings.sort_values(by="rating", ascending=False).head(5)
                for index, row in top5.iterrows():
                    movie_id = row["movieid"]
                    movie_details = query_movie_details(movie_id)
                    if movie_details is not None and not movie_details.empty:
                        title = movie_details.iloc[0]['title']
                        genres = movie_details.iloc[0]['genres']
                    else:
                        title, genres = "未知", "未知"
                    print(f"电影ID: {movie_id} | 标题: {title} | 类型: {genres} | 用户评分: {row['rating']}")
            else:
                print("该用户在数据库中暂无评分记录。")
                
        except ValueError:
            print("无效的用户ID，请输入数字。")
        except Exception as e:
            print(f"发生错误: {e}")