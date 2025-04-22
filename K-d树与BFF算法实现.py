import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree as SklearnKDTree
from collections import defaultdict
import psutil
import os

class KDNode:
    def __init__(self, point=None, split_dim=None, left=None, right=None, is_leaf=False, points=None):
        self.point = point            # 划分点
        self.split_dim = split_dim    # 划分维度
        self.left = left              # 左子树
        self.right = right            # 右子树
        self.is_leaf = is_leaf        # 是否为叶节点
        self.points = points          # 叶节点中的点集合

class KDTree:
    def __init__(self, points, leaf_size=10):
        """
        构建k-d树
        
        参数:
            points: 数据点集合，形状为(n, d)
            leaf_size: 叶节点中的最大点数
        """
        self.points = np.asarray(points)
        self.n, self.d = self.points.shape
        self.leaf_size = leaf_size
        self.root = self._build(np.arange(self.n))
        
    def _build(self, indices):
        """递归构建k-d树"""
        if len(indices) <= self.leaf_size:
            # 创建叶节点
            return KDNode(is_leaf=True, points=self.points[indices])
        
        # 选择具有最大方差的维度作为分割维度
        data = self.points[indices]
        split_dim = np.argmax(np.var(data, axis=0))
        
        # 按分割维度的中位数对点进行排序
        sorted_indices = indices[np.argsort(data[:, split_dim])]
        median_idx = len(sorted_indices) // 2
        
        # 创建内部节点
        node = KDNode(
            point=self.points[sorted_indices[median_idx]],
            split_dim=split_dim,
            left=self._build(sorted_indices[:median_idx]),
            right=self._build(sorted_indices[median_idx+1:])
        )
        return node
    
    def standard_search(self, query_point):
        """标准k-d树搜索"""
        query_point = np.asarray(query_point)
        best_dist = float('inf')
        best_point = None
        
        def search(node):
            nonlocal best_dist, best_point
            
            if node.is_leaf:
                # 在叶节点中搜索最近点
                for point in node.points:
                    dist = np.sum((point - query_point) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_point = point
                return
            
            # 决定优先搜索哪个子树
            split_val = node.point[node.split_dim]
            query_val = query_point[node.split_dim]
            
            if query_val <= split_val:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left
            
            # 先搜索更可能包含最近邻的子树
            search(first)
            
            # 计算查询点到分割超平面的距离
            dist_to_plane = (query_val - split_val) ** 2
            
            # 如果有可能在另一侧子树中找到更近的点，则搜索另一侧
            if dist_to_plane < best_dist:
                search(second)
        
        search(self.root)
        return best_point, np.sqrt(best_dist)
    
    def bbf_search(self, query_point, max_nodes=200):
        """
        BBF算法实现
        
        参数:
            query_point: 查询点
            max_nodes: 最大搜索节点数
        """
        query_point = np.asarray(query_point)
        best_dist = float('inf')
        best_point = None
        
        # 优先队列（最小堆），存储(距离, 节点)对
        # 距离的负值作为优先级（最大堆模拟）
        pq = [(-0, self.root)]  # (负优先级, 节点)
        nodes_visited = 0
        
        while pq and nodes_visited < max_nodes:
            # 弹出优先级最高的节点（距离最小）
            _, node = heapq.heappop(pq)
            
            if node.is_leaf:
                nodes_visited += 1
                # 检查叶节点中的所有点
                for point in node.points:
                    dist = np.sum((point - query_point) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_point = point
            else:
                # 计算查询点与分割维度的距离
                split_val = node.point[node.split_dim]
                query_val = query_point[node.split_dim]
                dist_to_plane = (query_val - split_val) ** 2
                
                # 决定先搜索哪个子树
                if query_val <= split_val:
                    near, far = node.left, node.right
                else:
                    near, far = node.right, node.left
                
                # 添加远端子树到优先队列（如果距离小于当前最佳距离）
                # 优先级是距离的倒数
                if far is not None:
                    priority = 1.0 / (dist_to_plane + 1e-10)  # 避免除零
                    heapq.heappush(pq, (-priority, far))
                
                # 添加近端子树到优先队列
                if near is not None:
                    # 近端子树优先级更高
                    heapq.heappush(pq, (-float('inf'), near))
        
        return best_point, np.sqrt(best_dist)

def brute_force_search(points, query_point):
    """暴力搜索算法"""
    points = np.asarray(points)
    query_point = np.asarray(query_point)
    
    # 计算查询点到所有点的距离
    distances = np.sum((points - query_point) ** 2, axis=1)
    nearest_idx = np.argmin(distances)
    
    return points[nearest_idx], np.sqrt(distances[nearest_idx])

def generate_random_data(n, d):
    """生成随机数据集"""
    return np.random.rand(n, d)

def evaluate_algorithms(dimensions, num_points=10000, num_queries=100, bbf_max_nodes=200):
    """评估不同算法在不同维度数据上的性能"""
    results = []
    
    for d in dimensions:
        print(f"评估 {d} 维数据...")
        
        # 生成数据集
        dataset = generate_random_data(num_points, d)
        queries = generate_random_data(num_queries, d)
        
        # 构建k-d树
        start_time = time.time()
        kdtree = KDTree(dataset)
        build_time = time.time() - start_time
        
        # 记录内存使用
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # 暴力搜索
        bf_times = []
        for query in queries:
            start_time = time.time()
            bf_result, bf_dist = brute_force_search(dataset, query)
            bf_times.append(time.time() - start_time)
        
        # 标准k-d树搜索
        kd_times = []
        kd_accuracy = []
        for query in queries:
            # 获取暴力搜索的真实最近邻
            true_nn, true_dist = brute_force_search(dataset, query)
            
            start_time = time.time()
            kd_result, kd_dist = kdtree.standard_search(query)
            kd_times.append(time.time() - start_time)
            
            # 计算准确率（距离比值）
            kd_accuracy.append(true_dist / kd_dist if kd_dist > 0 else 1.0)
        
        # BBF搜索
        bbf_times = []
        bbf_accuracy = []
        for query in queries:
            # 获取暴力搜索的真实最近邻
            true_nn, true_dist = brute_force_search(dataset, query)
            
            start_time = time.time()
            bbf_result, bbf_dist = kdtree.bbf_search(query, bbf_max_nodes)
            bbf_times.append(time.time() - start_time)
            
            # 计算准确率（距离比值）
            bbf_accuracy.append(true_dist / bbf_dist if bbf_dist > 0 else 1.0)
        
        # 计算成功率（距离比值≤1.05视为成功）
        kd_success_rate = np.mean([1 if ratio >= 0.95 else 0 for ratio in kd_accuracy])
        bbf_success_rate = np.mean([1 if ratio >= 0.95 else 0 for ratio in bbf_accuracy])
        
        # 记录结果
        results.append({
            'dimension': d,
            'bf_time': np.mean(bf_times),
            'kd_time': np.mean(kd_times),
            'bbf_time': np.mean(bbf_times),
            'kd_accuracy': np.mean(kd_accuracy),
            'bbf_accuracy': np.mean(bbf_accuracy),
            'kd_success_rate': kd_success_rate,
            'bbf_success_rate': bbf_success_rate,
            'memory_usage': memory_usage,
            'build_time': build_time
        })
    
    return results

def plot_results(results):
    """绘制实验结果"""
    dimensions = [r['dimension'] for r in results]
    
    # 查询时间对比
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(dimensions, [r['bf_time'] * 1000 for r in results], 'o-', label='暴力搜索')
    plt.plot(dimensions, [r['kd_time'] * 1000 for r in results], 's-', label='标准k-d树')
    plt.plot(dimensions, [r['bbf_time'] * 1000 for r in results], '^-', label='BBF搜索')
    plt.xlabel('维度')
    plt.ylabel('平均查询时间 (ms)')
    plt.title('查询时间vs维度')
    plt.legend()
    plt.grid(True)
    
    # 准确率对比
    plt.subplot(2, 2, 2)
    plt.plot(dimensions, [r['kd_accuracy'] for r in results], 's-', label='标准k-d树')
    plt.plot(dimensions, [r['bbf_accuracy'] for r in results], '^-', label='BBF搜索')
    plt.xlabel('维度')
    plt.ylabel('准确率 (真实距离/返回距离)')
    plt.title('准确率vs维度')
    plt.legend()
    plt.grid(True)
    
    # 成功率对比
    plt.subplot(2, 2, 3)
    plt.plot(dimensions, [r['kd_success_rate'] * 100 for r in results], 's-', label='标准k-d树')
    plt.plot(dimensions, [r['bbf_success_rate'] * 100 for r in results], '^-', label='BBF搜索')
    plt.xlabel('维度')
    plt.ylabel('成功率 (%)')
    plt.title('成功率vs维度')
    plt.legend()
    plt.grid(True)
    
    # 内存使用对比
    plt.subplot(2, 2, 4)
    plt.plot(dimensions, [r['memory_usage'] for r in results], 'o-')
    plt.xlabel('维度')
    plt.ylabel('内存使用 (MB)')
    plt.title('内存使用vs维度')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('bbf_comparison_results.png')
    plt.show()

if __name__ == "__main__":
    # 评估不同维度的数据
    dimensions = [2, 4, 8, 16, 32, 64, 128]
    results = evaluate_algorithms(dimensions)
    
    # 打印结果表格
    print("\n结果表格：")
    print("-" * 100)
    print(f"{'维度':<10}{'暴力搜索(ms)':<15}{'标准k-d树(ms)':<15}{'BBF搜索(ms)':<15}{'k-d树准确率':<15}{'BBF准确率':<15}{'k-d树成功率':<15}{'BBF成功率':<15}{'内存(MB)':<15}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['dimension']:<10}{r['bf_time']*1000:<15.4f}{r['kd_time']*1000:<15.4f}{r['bbf_time']*1000:<15.4f}{r['kd_accuracy']:<15.4f}{r['bbf_accuracy']:<15.4f}{r['kd_success_rate']*100:<15.2f}{r['bbf_success_rate']*100:<15.2f}{r['memory_usage']:<15.2f}")
    
    # 绘制结果图表
    plot_results(results)