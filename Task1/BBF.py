import numpy as np
import heapq
from KDtree import KDTree, KDNode

class BBFKDTree(KDTree):
    """BBF (Best Bin First) 搜索算法实现"""
    
    def __init__(self, points):
        """
        初始化BBF KD树
        :param points: 数据点集合，numpy数组格式，shape为(n, d)
        """
        super().__init__(points)  # 调用父类构造函数，构建标准k-d树
    
    def bbf_search(self, query_point, t=200):
        """
        BBF（Best Bin First）最近邻搜索算法
        :param query_point: 查询点
        :param t: 最大搜索叶子节点数
        :return: 最近邻点及距离
        """
        best_point = None
        best_dist = float('inf')
        
        # 使用优先队列管理搜索路径
        # 队列元素为 (优先级, 节点, 深度)
        # Python的heapq是最小堆，使用负优先级实现最大堆
        priority_queue = [(0, self.root, 0)]  # (优先级, 节点, 深度)
        
        # 计数已搜索的叶子节点
        searched_leaves = 0
        
        while priority_queue and searched_leaves < t:
            # 从优先队列中取出优先级最高的节点
            priority, node, depth = heapq.heappop(priority_queue)
            
            if node is None:
                continue
                
            # 如果节点是叶子节点（没有子节点），增加计数
            if node.left is None and node.right is None:
                searched_leaves += 1
                
            # 计算当前节点与查询点的距离
            point = node.point
            dist = self._euclidean_distance(query_point, point)
            
            # 更新最近邻
            if dist < best_dist:
                best_dist = dist
                best_point = point
            
            # 计算查询点到分割超平面的距离
            axis = node.axis
            plane_dist = abs(query_point[axis] - point[axis])
            
            # 确定优先搜索的子树
            if query_point[axis] < point[axis]:
                nearer_child = node.left
                farther_child = node.right
            else:
                nearer_child = node.right
                farther_child = node.left
            
            # 将远端子树加入优先队列，优先级为平面距离的倒数
            if farther_child is not None:
                # 距离越小，优先级越高（-1/distance)
                # 使用负值是因为Python的heapq是最小堆，我们需要模拟最大堆行为
                priority = -1.0 / max(plane_dist, 1e-10)  # 避免除零错误
                heapq.heappush(priority_queue, (priority, farther_child, depth + 1))
            
            # 将近端子树加入优先队列，优先级最高(-∞确保最高优先级)
            # 这确保了我们总是先搜索更可能包含最近邻的近端子树
            if nearer_child is not None:
                heapq.heappush(priority_queue, (-float('inf'), nearer_child, depth + 1))
        
        return best_point, best_dist
    
    def bbf_k_nearest_neighbors(self, query_point, k=1, t=200):
        """
        BBF k近邻搜索
        :param query_point: 查询点
        :param k: 返回的近邻数
        :param t: 最大搜索叶子节点数
        :return: k个最近邻点及其距离，按距离排序
        """
        if k <= 0:
            return []
        
        # 使用最大堆保存k个最近邻点
        nearest_neighbors = []  # (距离, 点)
        
        # 使用优先队列管理搜索路径，按距离分割超平面的距离排序（越小优先级越高）
        priority_queue = [(0, self.root, 0)]  # (优先级, 节点, 深度)
        
        # 计数已搜索的叶子节点
        searched_leaves = 0
        
        while priority_queue and searched_leaves < t:
            # 取出优先级最高的节点
            priority, node, depth = heapq.heappop(priority_queue)
            
            if node is None:
                continue
                
            # 如果节点是叶子节点，增加计数
            if node.left is None and node.right is None:
                searched_leaves += 1
                
            # 计算当前节点与查询点的距离
            point = node.point
            dist = self._euclidean_distance(query_point, point)
            
            # 更新最近邻
            if len(nearest_neighbors) < k:
                heapq.heappush(nearest_neighbors, (-dist, point))  # 使用负距离实现最大堆
            elif -dist > nearest_neighbors[0][0]:  # 如果当前点比堆顶点更近
                heapq.heappushpop(nearest_neighbors, (-dist, point))
            
            # 计算查询点到分割超平面的距离
            axis = node.axis
            plane_dist = abs(query_point[axis] - point[axis])
            
            # 确定优先搜索的子树
            if query_point[axis] < point[axis]:
                nearer_child = node.left
                farther_child = node.right
            else:
                nearer_child = node.right
                farther_child = node.left
            
            # 当前k个最近邻中的最大距离（如果还没找到k个，则为无穷大）
            current_max_dist = -nearest_neighbors[0][0] if nearest_neighbors else float('inf')
            
            # 将远端子树加入优先队列（如果可能包含更近的点）
            if farther_child is not None and (len(nearest_neighbors) < k or plane_dist < current_max_dist):
                priority = -1.0 / max(plane_dist, 1e-10)  # 避免除零错误
                heapq.heappush(priority_queue, (priority, farther_child, depth + 1))
            
            # 将近端子树加入优先队列，优先级最高
            if nearer_child is not None:
                heapq.heappush(priority_queue, (-float('inf'), nearer_child, depth + 1))
        
        # 返回结果：将距离转为正数，并按距离排序
        result = [(point, -dist) for dist, point in nearest_neighbors]
        result.sort(key=lambda x: x[1])
        return result


# 测试代码
if __name__ == "__main__":
    # 创建一些随机点
    np.random.seed(0)
    points = np.random.rand(1000, 3)  # 1000个3维点
    
    # 构建BBF KD树
    bbf_tree = BBFKDTree(points)
    
    # 随机选择一个查询点
    query = np.random.rand(3)
    
    # 测试不同的t值
    t_values = [10, 50, 100, 200, 500, 1000]
    
    print("查询点:", query)
    print("\n单点最近邻搜索:")
    print("t\t最近点\t\t距离")
    
    # 使用标准KD树搜索作为基准
    kdtree = KDTree(points)
    standard_nearest, standard_dist = kdtree.nearest_neighbor(query)
    print(f"标准KD树\t{standard_nearest}\t{standard_dist}")
    
    for t in t_values:
        bbf_nearest, bbf_dist = bbf_tree.bbf_search(query, t=t)
        print(f"{t}\t{bbf_nearest}\t{bbf_dist}")
    
    # k近邻测试
    k = 5
    print(f"\n{k}个最近邻 (t=200):")
    k_nearest = bbf_tree.bbf_k_nearest_neighbors(query, k=k, t=200)
    for i, (point, dist) in enumerate(k_nearest):
        print(f"{i+1}. 点: {point}, 距离: {dist}")
