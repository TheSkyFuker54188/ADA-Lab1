import os
import cv2
import numpy as np
import time
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

class HierarchicalClusteringImageRetrieval:
    """基于层次聚类树的图像检索系统"""
    
    def __init__(self, dataset_path, vocab_size=200, feature_type='SIFT', max_features=500):
        """
        初始化图像检索系统
        :param dataset_path: 图像数据集路径
        :param vocab_size: 视觉词典大小
        :param feature_type: 特征提取类型，'SIFT'或'ORB'
        :param max_features: 每张图像最多提取的特征点数量
        """
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size
        self.feature_type = feature_type
        self.max_features = max_features
        
        # 存储数据
        self.image_paths = []  # 图像路径列表
        self.image_features = {}  # 图像特征字典 {image_path: features}
        self.vocab = None  # 视觉词典
        self.vocab_tree = None  # 视觉词典KD树
        self.database_bows = []  # 数据库中所有图像的词袋表示
    
    def extract_features(self, image_path):
        """
        提取图像特征
        :param image_path: 图像路径
        :return: 特征描述符
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 根据选择的特征类型提取特征
        if self.feature_type == 'SIFT':
            # 使用SIFT提取特征
            detector = cv2.SIFT_create(nfeatures=self.max_features)
        else:
            # 使用ORB提取特征
            detector = cv2.ORB_create(nfeatures=self.max_features)
        
        # 检测关键点并计算描述符
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        
        return descriptors

    def build_vocabulary(self, sample_ratio=0.3, random_state=42):
        """
        构建视觉词典
        :param sample_ratio: 用于构建词典的图像采样比例
        :param random_state: 随机种子
        """
        print("开始构建视觉词典...")
        np.random.seed(random_state)
        
        # 获取所有图像路径
        self.image_paths = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"在数据集中找到 {len(self.image_paths)} 张图像")
        
        # 随机采样部分图像用于构建词典
        n_samples = max(int(len(self.image_paths) * sample_ratio), 1)
        sample_paths = np.random.choice(self.image_paths, n_samples, replace=False)
        
        print(f"使用 {n_samples} 张图像进行词典构建")
        
        # 提取特征
        all_descriptors = []
        for img_path in tqdm(sample_paths, desc="提取特征"):
            descriptors = self.extract_features(img_path)
            if descriptors is not None and descriptors.shape[0] > 0:
                # 存储提取的特征
                self.image_features[img_path] = descriptors
                all_descriptors.append(descriptors)
        
        if not all_descriptors:
            raise ValueError("无法从数据集中提取特征")
        
        # 合并所有描述符
        print("合并所有特征描述符...")
        all_descriptors = np.vstack(all_descriptors)
        print(f"特征描述符总数: {all_descriptors.shape[0]}")
        
        # 如果特征过多，随机采样
        if all_descriptors.shape[0] > 100000:
            print(f"特征数量太多，进行随机采样...")
            indices = np.random.choice(all_descriptors.shape[0], 100000, replace=False)
            all_descriptors = all_descriptors[indices]
            print(f"采样后的特征数量: {all_descriptors.shape[0]}")
        
        # 使用层次聚类
        print(f"使用层次聚类方法生成 {self.vocab_size} 个视觉单词...")
        start_time = time.time()
        # 使用Ward方法进行层次聚类
        Z = linkage(all_descriptors, method='ward', metric='euclidean')
        
        # 从层次结构中提取簇
        cluster_labels = fcluster(Z, self.vocab_size, criterion='maxclust')
        print(f"聚类完成，用时: {time.time() - start_time:.2f}秒")
        
        # 计算每个簇的中心作为视觉单词
        self.vocab = np.zeros((self.vocab_size, all_descriptors.shape[1]))
        for i in range(1, self.vocab_size + 1):
            cluster_points = all_descriptors[cluster_labels == i]
            if len(cluster_points) > 0:
                self.vocab[i-1] = np.mean(cluster_points, axis=0)
        
        # 构建KD树用于快速查找最近视觉单词
        self.vocab_tree = KDTree(self.vocab)
        
        print("视觉词典构建完成")

    def build_database(self):
        """
        为数据集中的所有图像构建词袋表示
        """
        print("为数据库图像生成词袋表示...")
        self.database_bows = []
        
        # 处理数据集中的每张图像
        for img_path in tqdm(self.image_paths, desc="处理图像"):
            # 如果特征已经提取过，直接使用
            if img_path in self.image_features:
                descriptors = self.image_features[img_path]
            else:
                # 否则提取特征
                descriptors = self.extract_features(img_path)
                if descriptors is not None:
                    self.image_features[img_path] = descriptors
            
            # 计算图像的词袋表示
            bow = self.compute_bow(descriptors)
            self.database_bows.append(bow)
        
        # 转换为NumPy数组便于后续处理
        self.database_bows = np.array(self.database_bows)
        
        print(f"数据库构建完成，包含 {len(self.database_bows)} 个词袋向量")

    def compute_bow(self, descriptors):
        """
        计算图像的词袋表示
        :param descriptors: 特征描述符
        :return: 词袋向量 (归一化后)
        """
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocab_size)
        
        # 为每个特征描述符找到最近的视觉单词
        distances, indices = self.vocab_tree.query(descriptors, k=1)
        indices = indices.flatten()
        
        # 计算词频直方图
        bow = np.zeros(self.vocab_size)
        for idx in indices:
            bow[idx] += 1
        
        # L2归一化
        norm = np.linalg.norm(bow)
        if norm > 0:
            bow = bow / norm
        
        return bow

    def query(self, query_path, top_k=10):
        """
        查询相似图像
        :param query_path: 查询图像路径
        :param top_k: 返回结果数量
        :return: [(图像路径, 相似度分数)]
        """
        print(f"查询图像: {query_path}")
        
        # 提取查询图像特征
        start_time = time.time()
        query_descriptors = self.extract_features(query_path)
        
        if query_descriptors is None:
            print("无法提取查询图像特征")
            return []
        
        # 计算查询图像的词袋表示
        query_bow = self.compute_bow(query_descriptors)
        
        # 计算查询向量与数据库中所有向量的余弦相似度
        similarities = cosine_similarity([query_bow], self.database_bows)[0]
        
        # 按相似度降序排序获取前top_k个结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 构建结果列表
        results = [(self.image_paths[idx], similarities[idx]) for idx in top_indices]
        
        query_time = time.time() - start_time
        print(f"查询完成，耗时: {query_time:.3f}秒")
        
        return results

    def display_results(self, query_path, results, save_path=None):
        """
        显示查询结果
        :param query_path: 查询图像路径
        :param results: 查询结果 [(图像路径, 相似度)]
        :param save_path: 保存结果图像的路径
        """
        # 计算子图数量和布局
        n_results = len(results)
        n_cols = min(5, n_results + 1)
        n_rows = (n_results + n_cols) // n_cols
        
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        
        # 显示查询图像
        plt.subplot(n_rows, n_cols, 1)
        query_img = cv2.imread(query_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        plt.imshow(query_img)
        plt.title("查询图像")
        plt.axis('off')
        
        # 显示结果图像
        for i, (img_path, similarity) in enumerate(results):
            plt.subplot(n_rows, n_cols, i + 2)  # +2 是因为第一张是查询图像
            result_img = cv2.imread(img_path)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            plt.imshow(result_img)
            plt.title(f"相似度: {similarity:.3f}")
            plt.axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"结果已保存至: {save_path}")
        
        plt.show()

    def save_model(self, filepath):
        """
        保存模型
        :param filepath: 保存路径
        """
        model_data = {
            'vocab': self.vocab,
            'image_paths': self.image_paths,
            'database_bows': self.database_bows,
            'vocab_size': self.vocab_size,
            'feature_type': self.feature_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存至: {filepath}")

    def load_model(self, filepath):
        """
        加载模型
        :param filepath: 模型路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab = model_data['vocab']
        self.image_paths = model_data['image_paths']
        self.database_bows = model_data['database_bows']
        self.vocab_size = model_data['vocab_size']
        self.feature_type = model_data['feature_type']
        
        # 重新构建KD树
        if self.vocab is not None:
            self.vocab_tree = KDTree(self.vocab)
        
        print(f"模型已从 {filepath} 加载")

# 测试代码
if __name__ == "__main__":
    # 系统参数
    dataset_path = './image'  # 图像数据集路径
    model_path = 'image_retrieval_model.pkl'  # 模型保存路径
    
    # 创建检索系统
    retrieval_system = HierarchicalClusteringImageRetrieval(
        dataset_path=dataset_path,
        vocab_size=200,  # 词典大小
        feature_type='SIFT',  # 特征类型: 'SIFT' 或 'ORB'
        max_features=500  # 每张图像最大特征点数
    )
    
    # 检查是否已有保存的模型
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        retrieval_system.load_model(model_path)
    else:
        # 构建系统
        print("构建新模型...")
        retrieval_system.build_vocabulary()
        retrieval_system.build_database()
        
        # 保存模型以便下次使用
        retrieval_system.save_model(model_path)
    
    # 随机选择一张图像作为查询图像
    query_path = np.random.choice(retrieval_system.image_paths)
    
    # 执行查询
    results = retrieval_system.query(query_path, top_k=10)
    
    # 显示结果
    retrieval_system.display_results(query_path, results, save_path='query_result.png')
