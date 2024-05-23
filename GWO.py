import random
import numpy

def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # 初始化 alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)  # 初始化Alpha狼位置，长度为dim
    Alpha_score = float("inf")  # 初始化Alpha狼得分，初始化为正无穷

    Beta_pos = numpy.zeros(dim)  # 初始化Beta狼位置，长度为dim
    Beta_score = float("inf")  # 初始化Beta狼得分，初始化为正无穷

    Delta_pos = numpy.zeros(dim)  # 初始化Delta狼位置，长度为dim
    Delta_score = float("inf")  # 初始化Delta狼得分，初始化为正无穷

    # 确保上下界是一个列表，长度为dim
    if not isinstance(lb, list):  
        lb = [lb] * dim  
    if not isinstance(ub, list):
        ub = [ub] * dim

    # 初始化所有狼的位置
    Positions = numpy.zeros((SearchAgents_no, dim))  # Positions矩阵形状为(SearchAgents_no,dim)
    for i in range(dim):
        # 将第i维位置初始化为SearchAgents_no个[0,1)的随机数乘上ub[i]-lb[i]，然后加上lb[i]
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    Convergence_curve = numpy.zeros(Max_iter)  # 初始化迭代结果

    #迭代寻优
    for l in range(0, Max_iter):
        # 对每个狼进行操作
        for i in range(0, SearchAgents_no):
            # 将狼的位置限制在上下界范围内
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            # 计算狼的目标函数得分
            fitness = objf(Positions[i, :])

            # 更新Alpha, Beta, 和 Delta
            if fitness < Alpha_score:
                Alpha_score = fitness  
                Alpha_pos = Positions[i, :].copy()  

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  
                Beta_pos = Positions[i, :].copy() 

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  
                Delta_pos = Positions[i, :].copy()  

        a = 2 - l * ((2) / Max_iter);  # 计算a参数

        # 对每个狼进行操作
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # 生成[0,1)的随机数
                r2 = random.random()  # 生成[0,1)的随机数

                A1 = 2 * a * r1 - a;  # 算法中的公式
                C1 = 2 * r2;  

                # 计算与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                r1 = random.random()  # 生成[0,1)的随机数
                r2 = random.random()  # 生成[0,1)的随机数

                A2 = 2 * a * r1 - a;  # 算法中的公式
                C2 = 2 * r2;

                # 计算与Beta狼的距离
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;

                r1 = random.random()  # 生成[0,1)的随机数
                r2 = random.random()  # 生成[0,1)的随机数

                A3 = 2 * a * r1 - a;  # 算法中的公式
                C3 = 2 * r2;

                # 计算与Delta狼的距离
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;

                # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。
                Positions[i, j] = (X1 + X2 + X3) / 3  

        Convergence_curve[l] = Alpha_score;  # 存储每一次迭代结果

        if (l % 1 == 0):
            print(['迭代次数为' + str(l) + ' 的迭代结果' + str(Alpha_score)]);  # 每一次迭代都输出结果

# 定义目标函数
def F1(x):
    s = numpy.sum(x**2);
    return s

# 主程序
func_details = ['F1', -100, 100, 30]  
function_name = func_details[0]  
Max_iter = 1000  # 迭代次数
lb = -100  # 下界
ub = 100  # 上届
dim = 30  # 狼的寻值范围
SearchAgents_no = 5  # 寻值的狼的数量
x = GWO(F1, lb, ub, dim, SearchAgents_no, Max_iter)  # 调用GWO函数开始优化
