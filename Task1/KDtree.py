import numpy as np
from collections import namedtuple

class KDNode:
    """k-d树的节点类"""
    def __init__(self, point=None, axis=None, left=None, right=None):
        self.point = point  # 节点存储的点
        self.axis = axis    # 分割维度
        self.left = left    # 左子树
        self.right = right  # 右子树

class KDTree:
    """k-d树实现类"""
    def __init__(self, points):
        """
        构建KD树
        :param points: 数据点集合，numpy数组格式，shape为(n, d)
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        self.points = points
        self.n, self.d = points.shape  # 点的数量和维度
        self.root = self._build_tree(points, 0)
    
    def _build_tree(self, points, depth):
        """
        递归构建KD树
        :param points: 当前节点包含的点集
        :param depth: 当前深度
        :return: 节点
        """
        if len(points) == 0:
            return None
        
        # 确定分割的维度 (轮流使用每个维度)
        axis = depth % self.d
        
        # 按当前维度排序并选取中位数作为分割点
        sorted_indices = np.argsort(points[:, axis])
        median_idx = len(points) // 2
        median_point_idx = sorted_indices[median_idx]
        
        # 创建节点
        node = KDNode(
            point=points[median_point_idx],
            axis=axis,
            left=self._build_tree(points[sorted_indices[:median_idx]], depth + 1),
            right=self._build_tree(points[sorted_indices[median_idx+1:]], depth + 1)
        )
        
        return node
    
    def _euclidean_distance(self, point1, point2):
        """
        计算两点间的欧氏距离
        :param point1: 第一个点
        :param point2: 第二个点
        :return: 欧氏距离
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def nearest_neighbor(self, query_point):
        """
        找出离查询点最近的点（标准k-d树搜索）
        :param query_point: 查询点
        :return: 最近点和距离
        """
        best_point = None
        best_dist = float('inf')
        
        def search(node, depth):
            nonlocal best_point, best_dist
            
            if node is None:
                return
            
            # 计算当前节点点到查询点的距离
            point = node.point
            dist = self._euclidean_distance(query_point, point)
            
            # 如果找到更近的点，更新结果
            if dist < best_dist:
                best_dist = dist
                best_point = point
            
            # 确定搜索顺序
            axis = depth % self.d
            next_branch = None
            other_branch = None
            
            # 决定先搜索哪个分支
            if query_point[axis] < point[axis]:
                next_branch = node.left
                other_branch = node.right
            else:
                next_branch = node.right
                other_branch = node.left
            
            # 递归搜索最可能包含最近点的子树
            search(next_branch, depth + 1)
            
            # 检查是否需要搜索另一个子树
            # 如果查询点到分割超平面的距离小于当前最近距离，则需要搜索另一个子树
            if abs(query_point[axis] - point[axis]) < best_dist:
                search(other_branch, depth + 1)
        
        # 从根节点开始搜索
        search(self.root, 0)
        
        return best_point, best_dist
    
    def k_nearest_neighbors(self, query_point, k=1):
        """
        找出离查询点最近的k个点
        :param query_point: 查询点
        :param k: 需要返回的最近邻点数量
        :return: k个最近点及其距离的列表，按距离排序
        """
        if k <= 0:
            return []
        
        # 存储k个最近邻 [(距离, 点)]
        import heapq
        nearest_neighbors = []
        
        def search(node, depth):
            if node is None:
                return
            
            # 计算当前节点点到查询点的距离
            point = node.point
            dist = self._euclidean_distance(query_point, point)
            
            # 如果还没找到k个点，或者当前点比堆顶点更近，则更新最近邻
            if len(nearest_neighbors) < k:
                heapq.heappush(nearest_neighbors, (-dist, point))  # 使用负距离实现最大堆
            elif -dist > nearest_neighbors[0][0]:  # 当前距离小于最大距离
                heapq.heappushpop(nearest_neighbors, (-dist, point))
            
            # 确定搜索顺序
            axis = depth % self.d
            next_branch = None
            other_branch = None
            
            # 决定先搜索哪个分支
            if query_point[axis] < point[axis]:
                next_branch = node.left
                other_branch = node.right
            else:
                next_branch = node.right
                other_branch = node.left
            
            # 递归搜索最可能包含最近点的子树
            search(next_branch, depth + 1)
            
            # 如果还没找到k个点，或者查询点到分割超平面的距离小于当前最大距离，则搜索另一个子树
            if len(nearest_neighbors) < k or abs(query_point[axis] - point[axis]) < -nearest_neighbors[0][0]:
                search(other_branch, depth + 1)
        
        # 从根节点开始搜索
        search(self.root, 0)
        
        # 返回结果，转换回正距离并按距离排序
        result = [(point, -dist) for dist, point in nearest_neighbors]
        result.sort(key=lambda x: x[1])  # 按距离排序
        return result


# 测试代码
if __name__ == "__main__":
    # 创建一些随机点
    np.random.seed(0)
    points = np.random.rand(1000, 3)  # 1000个3维点
    
    # 构建KD树
    kdtree = KDTree(points)
    
    # 随机选择一个查询点
    query = np.random.rand(3)
    
    # 找到最近邻
    nearest, dist = kdtree.nearest_neighbor(query)
    
    print("查询点:", query)
    print("最近邻点:", nearest)
    print("距离:", dist)
    
    # 找到k个最近邻
    k = 5
    k_nearest = kdtree.k_nearest_neighbors(query, k)
    
    print(f"\n{k}个最近邻:")
    for i, (point, dist) in enumerate(k_nearest):
        print(f"{i+1}. 点: {point}, 距离: {dist}")
