import numpy as np

def brute_force_search(points, query_point):
    """
    暴力搜索最近邻点
    :param points: 数据点集合，numpy数组格式，shape为(n, d)
    :param query_point: 查询点
    :return: 最近的点和距离
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    if not isinstance(query_point, np.ndarray):
        query_point = np.array(query_point)
    
    # 计算查询点与所有点的欧氏距离
    distances = np.sqrt(np.sum((points - query_point) ** 2, axis=1))
    
    # 找到最小距离的索引
    min_idx = np.argmin(distances)
    
    return points[min_idx], distances[min_idx]

def brute_force_knn(points, query_point, k=1):
    """
    暴力搜索k近邻点
    :param points: 数据点集合
    :param query_point: 查询点
    :param k: 需要返回的近邻数量
    :return: k个最近点及其距离的列表，按距离排序
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    if not isinstance(query_point, np.ndarray):
        query_point = np.array(query_point)
    
    # 计算查询点与所有点的欧氏距离
    distances = np.sqrt(np.sum((points - query_point) ** 2, axis=1))
    
    # 找到k个最小距离的索引
    k = min(k, len(points))  # 确保k不超过点的数量
    nearest_indices = np.argsort(distances)[:k]
    
    # 返回结果
    return [(points[i], distances[i]) for i in nearest_indices]


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    points = np.random.rand(1000, 3)  # 1000个3维点
    query = np.random.rand(3)
    
    nearest, dist = brute_force_search(points, query)
    print(f"查询点: {query}")
    print(f"最近点: {nearest}")
    print(f"距离: {dist}")
    
    k = 5
    k_nearest = brute_force_knn(points, query, k)
    print(f"\n{k}个最近邻:")
    for i, (point, dist) in enumerate(k_nearest):
        print(f"{i+1}. 点: {point}, 距离: {dist}")
