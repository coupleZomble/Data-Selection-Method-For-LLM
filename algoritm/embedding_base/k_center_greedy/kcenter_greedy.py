import numpy as np

def split_simliarity_matrix(train_embedding_matrix, split_size):
    block_num, last_matrix_size = np.divmod(train_embedding_matrix.shape[0], split_size)
    split_matrix_list = []
    for i in range(block_num):
        tmp = np.matmul(train_embedding_matrix \
                        ,train_embedding_matrix[ int(i*split_size) :int((i+1)*split_size) ].T)
        split_matrix_list.append(tmp)
    tmp = np.matmul(train_embedding_matrix \
                        ,train_embedding_matrix[ int(block_num*split_size) :].T) 
    split_matrix_list.append(tmp)
    return split_matrix_list


def return_inital_point(split_sim_matrix,split_size):
    idx = 0
    min_value = np.inf
    for i in range(len(split_sim_matrix)):
        tmp = np.min(split_sim_matrix[i])
        if tmp < min_value:
            min_value = tmp
            idx = i
    target_matrix = split_sim_matrix[idx]
    r,c = np.divmod(np.argmin(target_matrix),split_size)
    return r


def get_similar(query_id, split_sim_matrix, pop_point_list=[]):
    # 提前创建布尔掩码 
    mask = np.ones(sum(matrix.shape[1] for matrix in split_sim_matrix), dtype=bool)
    mask[pop_point_list] = False
    
    # 拼接向量，并直接使用布尔索引
    sim_vector = np.concatenate(tuple(matrix[query_id] for matrix in split_sim_matrix) , axis=0)
    
    # 使用布尔索引来生成 key 和 value
    key = np.arange(len(sim_vector))[mask]
    value = sim_vector[mask]
    
    # 创建字典
    pair = dict(zip(key, value))
    return pair
# old version
def kcenter_greedy(split_sim_matrix, k=500, seed = 144, inital_point_id = None):
    if inital_point_id is None:
    # inital center data
        idxs = list(range( max(split_sim_matrix[0].shape) ))
        np.random.seed(seed)
        new_centers_id = np.random.choice(idxs)
        # 中心点的集合
    else:
        new_centers_id = inital_point_id
    centers_list = [new_centers_id] 
    distence_list = []

    old_distence_to_center = get_similar(new_centers_id, split_sim_matrix, pop_point_list=centers_list)
    
    # while len(centers_list) < k:
    for i in range(k-1):
        # 最低相似度的点
        new_centers_id = min(old_distence_to_center, key=old_distence_to_center.get)
        distence_list.append(old_distence_to_center[new_centers_id])
        print(f'{i}, {old_distence_to_center[new_centers_id]}')
        # if(old_distence_to_center[new_centers_id]>=0.5):
        #     break
        old_distence_to_center.pop(new_centers_id)
        
        centers_list.append(new_centers_id)
        new_distence_to_center = get_similar(new_centers_id, split_sim_matrix, pop_point_list=centers_list)
        # 更新其他点离中心点的距离 
        old_distence_to_center = {key: max(new_distence_to_center[key], old_distence_to_center[key]) for key in new_distence_to_center}
    print('finish')
    return centers_list,distence_list



def compute_sim_vector(center_id, split_sim_matrix, mask):
    """
    计算给定中心点 center_id 对所有点的相似度向量，并对已选点设置无效值 (-np.inf)
    """
    # 拼接各个矩阵中 center_id 对应的向量
    sim_vector = np.concatenate([matrix[center_id] for matrix in split_sim_matrix], axis=0)
    sim_vector[~mask] = -np.inf  # 已经选中的点不参与比较
    return sim_vector

# new version
def kcenter_greedy_optimized(split_sim_matrix, k=500, seed=144, inital_point_id=None):
    """
    优化版的 k-center 贪心算法：
    - 使用 NumPy 数组维护所有点的最佳相似度（即与任一中心点之间的最大相似度）。
    - 用布尔掩码记录哪些点已被选为中心。
    """
    # 计算总点数 N：所有矩阵列数之和
    N = sum(matrix.shape[1] for matrix in split_sim_matrix)
    
    # 初始化布尔掩码，True 表示该点未被选中
    mask = np.ones(N, dtype=bool)
    
    # 选取初始中心点
    if inital_point_id is None:
        # 这里假设第一个矩阵的行数可作为索引范围（可根据具体情况调整）
        idxs = np.arange(split_sim_matrix[0].shape[0])
        np.random.seed(seed)
        init_center = np.random.choice(idxs)
    else:
        init_center = inital_point_id
    
    centers_list = [init_center]
    mask[init_center] = False  # 标记已选中心点
    
    # 计算初始中心与所有点之间的相似度向量
    best_similarity = compute_sim_vector(init_center, split_sim_matrix, mask)
    distence_list = []
    
    for i in range(k - 1):
        # 在未选点中找出相似度最小（即离中心最远）的点
        valid_indices = np.where(mask)[0]
        if valid_indices.size == 0:
            break
        new_center = valid_indices[np.argmin(best_similarity[valid_indices])]
        dist = best_similarity[new_center]
        distence_list.append(dist)
        print(f'{i}, {dist}')
        
        # 更新中心点集合及掩码
        centers_list.append(new_center)
        mask[new_center] = False
        
        # 计算新中心与所有点的相似度向量
        new_sim = compute_sim_vector(new_center, split_sim_matrix, mask)
        # 更新所有未选点的最佳相似度：取当前值与新中心相似度的较大者
        best_similarity[mask] = np.maximum(best_similarity[mask], new_sim[mask])
    
    print('finish')
    return centers_list, distence_list


def main(train_embedding_matrix, split_size):
    split_sim_matrix = split_simliarity_matrix(train_embedding_matrix=train_embedding_matrix,split_size=split_size) 
    print('finish split')
    intial_point = return_inital_point(split_sim_matrix,split_size)
    print(f'finish find init point: {intial_point}')
    k_size = train_embedding_matrix.shape[0]
    kcenter_greedy_optimized(split_sim_matrix,
                   k=k_size,
                   seed = 144,
                   inital_point_id = intial_point)