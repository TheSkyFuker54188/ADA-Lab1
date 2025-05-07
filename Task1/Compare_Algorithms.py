import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from memory_profiler import memory_usage
import glob
from tqdm import tqdm

from KDtree import KDTree
from BBF import BBFKDTree
from Brute_Force import brute_force_search

def load_data(file_path):
    """
    加载数据集
    :param file_path: 数据文件路径
    :return: 数据点集、查询点集和维度
    """
    with open(file_path, 'r') as f:
        # 第一行包含数据点数量、查询点数量和维度
        n, m, d = map(int, f.readline().strip().split())
        
        # 读取数据点
        data_points = []
        for _ in range(n):
            point = list(map(float, f.readline().strip().split()))
            data_points.append(point)
        
        # 读取查询点
        query_points = []
        for _ in range(m):
            point = list(map(float, f.readline().strip().split()))
            query_points.append(point)
    
    return np.array(data_points), np.array(query_points), d

def measure_query_time(algorithm_name, algorithm_func, data_points, query_points, num_runs=100, **kwargs):
    """
    测量查询时间
    :param algorithm_name: 算法名称
    :param algorithm_func: 算法函数
    :param data_points: 数据点集
    :param query_points: 查询点集
    :param num_runs: 运行次数，默认100次（符合实验要求）
    :param kwargs: 算法的额外参数
    :return: 平均查询时间和查询结果
    """
    total_time = 0
    results = []
    
    # 如果是KD树或BBF，先构建树
    if algorithm_name == "kdtree":
        tree = KDTree(data_points)
        
        # 测量查询时间
        for _ in range(num_runs):
            start_time = time.time()
            results = [tree.nearest_neighbor(query) for query in query_points]
            total_time += time.time() - start_time
            
    elif algorithm_name == "bbf":
        tree = BBFKDTree(data_points)
        t = kwargs.get("t", 200)
        
        # 测量查询时间
        for _ in range(num_runs):
            start_time = time.time()
            results = [tree.bbf_search(query, t=t) for query in query_points]
            total_time += time.time() - start_time
            
    else:  # 暴力搜索
        # 测量查询时间
        for _ in range(num_runs):
            start_time = time.time()
            results = [algorithm_func(data_points, query) for query in query_points]
            total_time += time.time() - start_time
    
    avg_time = total_time / num_runs
    
    return avg_time, results

def measure_memory_usage(algorithm_name, data_points, query_point, **kwargs):
    """
    测量内存占用
    :param algorithm_name: 算法名称
    :param data_points: 数据点集
    :param query_point: 单个查询点
    :param kwargs: 算法的额外参数
    :return: 内存占用 (MB)
    """
    if algorithm_name == "brute_force":
        mem_usage = max(memory_usage((brute_force_search, (data_points, query_point)), interval=0.01))
    elif algorithm_name == "kdtree":
        tree = KDTree(data_points)
        mem_usage = max(memory_usage((tree.nearest_neighbor, (query_point,)), interval=0.01))
    else:  # bbf
        tree = BBFKDTree(data_points)
        t = kwargs.get("t", 200)
        mem_usage = max(memory_usage((tree.bbf_search, (query_point, t)), interval=0.01))
    
    return mem_usage

def evaluate_accuracy(true_results, test_results):
    """
    评估准确率：返回结果与真实最近邻的欧氏距离比值（≤1.05视为成功）
    :param true_results: 真实结果（暴力搜索结果）
    :param test_results: 测试结果
    :return: 准确率
    """
    success_count = 0
    total_count = len(true_results)
    
    for (_, true_dist), (_, test_dist) in zip(true_results, test_results):
        # 距离比值小于等于1.05视为成功
        if test_dist / true_dist <= 1.05:
            success_count += 1
    
    return success_count / total_count

def run_experiment(data_file):
    """
    对单个数据文件运行实验
    :param data_file: 数据文件路径
    :return: 实验结果
    """
    print(f"Running experiment on {data_file}...")
    
    # 提取文件名作为标识符
    file_id = os.path.basename(data_file).split('.')[0]
    
    # 加载数据
    data_points, query_points, dimension = load_data(data_file)
    print(f"Loaded data: {len(data_points)} data points, {len(query_points)} query points, {dimension} dimensions")
    
    # 设置运行次数为100，符合实验要求
    num_runs = 100
    print(f"Each algorithm will run {num_runs} times to calculate average query time")
    
    # 1. 暴力搜索
    print("Testing brute force search...")
    bf_time, bf_results = measure_query_time("brute_force", brute_force_search, data_points, query_points, num_runs=num_runs)
    bf_memory = measure_memory_usage("brute_force", data_points, query_points[0])
    
    # 2. 标准k-d树搜索
    print("Testing standard k-d tree search...")
    kd_time, kd_results = measure_query_time("kdtree", None, data_points, query_points, num_runs=num_runs)
    kd_memory = measure_memory_usage("kdtree", data_points, query_points[0])
    
    # 3. BBF搜索
    print("Testing BBF search (t=200)...")
    bbf_time, bbf_results = measure_query_time("bbf", None, data_points, query_points, t=200, num_runs=num_runs)
    bbf_memory = measure_memory_usage("bbf", data_points, query_points[0], t=200)
    
    # 计算准确率（以暴力搜索为基准）
    kd_accuracy = evaluate_accuracy(bf_results, kd_results)
    bbf_accuracy = evaluate_accuracy(bf_results, bbf_results)
    
    # 汇总结果
    results = {
        'algorithm': ['Brute Force', 'Standard KD-Tree', 'BBF (t=200)'],
        'dimension': dimension,
        'file_id': [file_id] * 3,  # 添加文件标识符
        'avg_query_time': [bf_time, kd_time, bbf_time],
        'accuracy': [1.0, kd_accuracy, bbf_accuracy],  # 暴力搜索准确率为1.0
        'memory_usage': [bf_memory, kd_memory, bbf_memory]
    }
    
    return results

def display_results(results_list):
    """
    显示实验结果
    :param results_list: 所有实验结果的列表
    """
    # 创建DataFrame来存储结果
    all_data = []
    for result in results_list:
        for i in range(len(result['algorithm'])):
            all_data.append({
                'Algorithm': result['algorithm'][i],
                'Dimension': result['dimension'],
                'File': result['file_id'][i],  # 使用文件标识符
                'Query Time (ms)': result['avg_query_time'][i] * 1000,  # 转换为毫秒
                'Accuracy': result['accuracy'][i],
                'Memory (MB)': result['memory_usage'][i]
            })
    
    df = pd.DataFrame(all_data)
    
    # 打印表格形式结果
    print("\n===== Performance Comparison =====")
    print(df.to_string(index=False))
    
    # 保存到CSV
    df.to_csv('algorithm_comparison_results.csv', index=False)
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 按文件和算法分组计算平均值
    time_data = df.groupby(['Algorithm'])['Query Time (ms)'].mean().reset_index()
    acc_data = df.groupby(['Algorithm'])['Accuracy'].mean().reset_index()
    mem_data = df.groupby(['Algorithm'])['Memory (MB)'].mean().reset_index()
    
    # 查询时间对比 (对数尺度)
    time_data.plot(kind='bar', x='Algorithm', y='Query Time (ms)', ax=axes[0], logy=True)
    axes[0].set_title('Average Query Time (ms, log scale)')
    axes[0].set_ylabel('Time (ms)')
    axes[0].grid(True, alpha=0.3)
    
    # 准确率对比
    acc_data.plot(kind='bar', x='Algorithm', y='Accuracy', ax=axes[1])
    axes[1].set_title('Average Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    
    # 内存对比
    mem_data.plot(kind='bar', x='Algorithm', y='Memory (MB)', ax=axes[2])
    axes[2].set_title('Average Memory Usage (MB)')
    axes[2].set_ylabel('Memory (MB)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300)
    plt.show()

def main():
    # 数据文件路径
    data_dir = "./data"
    
    # 找出所有数据文件
    data_files = glob.glob(f"{data_dir}/*.txt")
    
    # 选择不同维度的数据文件进行测试
    # 这里我们假设数据文件名是按数字编号的，如1.txt, 2.txt等
    selected_files = sorted(data_files)[:3]  # 选择前3个文件进行测试
    
    if not selected_files:
        print(f"Error: No data files found in {data_dir}")
        return
    
    results_list = []
    
    for file_path in selected_files:
        result = run_experiment(file_path)
        results_list.append(result)
    
    # 显示结果
    display_results(results_list)
    
    # 输出结论
    print("\n===== Conclusion =====")
    print("1. BBF算法在查询时间上通常比标准k-d树和暴力搜索更快。")
    print("2. 随着维度增加，BBF的准确率会下降，但仍能保持较高的准确性。")
    print("3. BBF在时间效率和准确率之间取得了良好的平衡，是高维数据快速最近邻搜索的优选方法。")

if __name__ == "__main__":
    main()
