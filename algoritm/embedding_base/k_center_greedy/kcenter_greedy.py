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


def main(train_embedding_matrix, split_size):
    split_sim_matrix = split_simliarity_matrix(train_embedding_matrix=train_embedding_matrix,split_size=split_size) 
    print('finish split')
    intial_point = return_inital_point(split_sim_matrix,split_size)
    print(f'finish find init point: {intial_point}')
    k_size = train_embedding_matrix.shape[0]
    kcenter_greedy(split_sim_matrix,
                   k=k_size,
                   seed = 144,
                   inital_point_id = intial_point)