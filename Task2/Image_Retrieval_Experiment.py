import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Image_Retrieval import HierarchicalClusteringImageRetrieval

def run_parameter_experiment(dataset_path, query_images, vocab_sizes, feature_types):
    """
    使用不同参数运行实验
    :param dataset_path: 数据集路径
    :param query_images: 查询图像列表
    :param vocab_sizes: 词典大小列表
    :param feature_types: 特征类型列表
    :return: 实验结果DataFrame
    """
    results = []
    
    for feature_type in feature_types:
        for vocab_size in vocab_sizes:
            print(f"\n===== 测试参数: feature_type={feature_type}, vocab_size={vocab_size} =====")
            
            # 创建并初始化系统
            model_path = f'model_{feature_type}_{vocab_size}.pkl'
            
            system = HierarchicalClusteringImageRetrieval(
                dataset_path=dataset_path,
                vocab_size=vocab_size,
                feature_type=feature_type,
                max_features=500
            )
            
            # 构建或加载模型
            if os.path.exists(model_path):
                system.load_model(model_path)
            else:
                # 记录构建时间
                build_start = time.time()
                system.build_vocabulary()
                system.build_database()
                build_time = time.time() - build_start
                system.save_model(model_path)
            
            # 对每个查询图像执行查询
            for query_path in query_images:
                # 记录查询时间
                query_start = time.time()
                query_results = system.query(query_path, top_k=10)
                query_time = time.time() - query_start
                
                # 保存结果图像
                result_img_path = f'result_{feature_type}_{vocab_size}_{os.path.basename(query_path)}.png'
                system.display_results(query_path, query_results, save_path=result_img_path)
                
                # 记录结果
                results.append({
                    'Feature Type': feature_type,
                    'Vocabulary Size': vocab_size,
                    'Query Image': os.path.basename(query_path),
                    'Query Time (s)': query_time,
                    'Top Match Similarity': query_results[0][1] if query_results else 0,
                    'Result Image': result_img_path
                })
    
    # 将结果转换为DataFrame
    return pd.DataFrame(results)

def analyze_results(results_df):
    """
    分析实验结果
    :param results_df: 实验结果DataFrame
    """
    # 按特征类型和词典大小分组计算平均查询时间
    avg_times = results_df.groupby(['Feature Type', 'Vocabulary Size'])['Query Time (s)'].mean().reset_index()
    
    # 绘制查询时间比较图
    plt.figure(figsize=(10, 6))
    
    for feature_type in avg_times['Feature Type'].unique():
        feature_data = avg_times[avg_times['Feature Type'] == feature_type]
        plt.plot(feature_data['Vocabulary Size'], feature_data['Query Time (s)'], 
                 marker='o', label=feature_type)
    
    plt.xlabel('词典大小')
    plt.ylabel('平均查询时间 (秒)')
    plt.title('不同参数下的查询性能比较')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('query_performance_comparison.png', dpi=300)
    plt.show()
    
    # 打印性能统计
    print("\n===== 性能统计 =====")
    print(avg_times)
    
    # 按特征类型和词典大小分组计算平均相似度
    avg_similarity = results_df.groupby(['Feature Type', 'Vocabulary Size'])['Top Match Similarity'].mean().reset_index()
    print("\n===== 平均相似度 =====")
    print(avg_similarity)

def main():
    # 设置参数
    dataset_path = './image'
    
    # 选择3张图像作为查询
    all_images = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                all_images.append(os.path.join(root, file))
    
    if len(all_images) < 3:
        print(f"警告: 只找到 {len(all_images)} 张图像")
        query_images = all_images
    else:
        # 随机选择3张图像
        np.random.seed(42)  # 固定随机种子以便重现结果
        query_images = np.random.choice(all_images, 3, replace=False)
    
    # 实验参数
    vocab_sizes = [100, 200, 500]
    feature_types = ['SIFT', 'ORB']
    
    # 运行实验
    print(f"使用 {len(query_images)} 张查询图像进行测试")
    results = run_parameter_experiment(dataset_path, query_images, vocab_sizes, feature_types)
    
    # 保存结果
    results.to_csv('experiment_results.csv', index=False)
    
    # 分析结果
    analyze_results(results)

if __name__ == "__main__":
    main()
