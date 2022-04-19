##环境设定
import numpy as np
import matplotlib.pyplot as plt
from deap import base,tools,creator,algorithms
import random

params = {
    'font.family':'serif',
    'figure.dpi':300,
    'savefig.dpi':300,
    'font.size':12,
    'legend.fontsize':'small'
}
plt.rcParams.update(params)
from copy import deepcopy
#--------------------------------
##问题定义
creator.create('FitnessMin',base.Fitness,weights=(-1,0,))#最小化问题
#给个体一个routes属性用来记录其表示路线
creator.create('Individual',list,fitness=creator.FitnessMin)

#--------------
##个体编码
#用字典存储所有参数--配送中心坐标、顾客坐标、顾客需求、到达时间窗口、服务时间、车型载重量
dataDict = {}
#节点坐标，节点0是配送中心得坐标
dataDict['NodeCoor'] = [(15,12),(3,13),(3,17),(6,18),(8,17),(10,14),
                        (14,13),(15,11),(15,15),(17,11),(17,16),
                        (18,19),(19,9),(19,21),(21,22),(23,9),
                        (23,22),(24,11),(27,21),(26,6),(26,9),
                        (27,2),(27,4),(27,17),(28,7),(29,14),
                        (29,18),(30,1),(30,8),(30,15),(30,17)]
# 将配送中心需求设置为0
dataDict['Demand'] = [0,50,50,60,30,90,10,20,10,30,20,30,10,10,10,
                      40,51,20,20,20,30,30,30,10,60,30,20,30,40,20,20]
dataDict['Maxload'] = 400
dataDict['SerciveTime'] = 1
def genInd(dataDict = dataDict):
    '''生成个体，对我们的问题来说，困难之处在于车辆数目是不定的'''
    nCustomer = len(dataDict['NodeCoor']) - 1 #顾客数量
    perm = np.random.permutation(nCustomer) + 1 #生成顾客的随机排列，注意顾客编号为1--n
    pointer = 0  # 迭代指针
    lowPointer = 0 #  指针指向下界
    permSlice = []
    # 当指针不指向序列末尾时
    while pointer < nCustomer -1:
        vehicleload = 0
        #  当不超载时，继续装载
        while (vehicleload < dataDict['Maxload']) and (pointer < nCustomer -1):
            vehicleload += dataDict['Demand'][perm[pointer]]
            pointer  += 1
        if lowPointer+1 < pointer:
            tempPointer = np.random.randint(lowPointer+1,pointer)
            permSlice.append(perm[lowPointer:tempPointer].tolist())
            lowPointer = tempPointer
            pointer = tempPointer
        else:
            permSlice.append(perm[lowPointer::].tolist())
            break
    #  将路线片段合并为染色体
    ind = [0]
    for  eachRoute in permSlice:
         ind = ind + eachRoute + [0]
    return ind
#-----------------------------------
## 评价函数
# 染色体编码
def decodeInd(ind):
    '''从染色体解码回路线片段，每条路径都是以0开头与结尾'''
    indCopy = np.array(deepcopy(ind)) # 复制ind，防止直接对染色体进行改动
    idxlist = list(range(len(indCopy)))
    zeroIdx = np.asarray(idxlist)[indCopy == 0]
    routes = []
    for i,j in zip(zeroIdx[0::],zeroIdx[1::]):
        routes.append(ind[i:j]+[0])
    return routes
def calDist(pos1,pos2):
    '''计算距离的辅助函数，根据给出的坐标pos1和pos2，返回两点之间的距离
    输入：pos1,pos2 -- (x,y)元组
    输出：欧几里得距离'''
    return np.sqrt((pos1[0] - pos2[0])*(pos1[0] - pos2[0]) + (pos1[1] - pos2[1])*(pos1[1] - pos2[1]))

#
def loadPenalty(routes):
    '''辅助函数，因为在交叉和突变中可能会产生不符合负载约束的个体，需要对不合要求的个体进行惩罚'''
    penalty = 0
    # 计算每条路径的负载，取max(0,routeload - maxload)计入惩罚项
    for eachRoute in routes:
        routeLoad = np.sum([dataDict['Demand'][i] for i in eachRoute])
        penalty += max(0,routeLoad - dataDict['Maxload'])
    return penalty

def calRoutelen(routes,dataDict=dataDict):
    '''辅助函数，返回给定路径的总长度'''
    totalDistance = 0 # 记录各条路线的总长度
    for eachRoute in routes:
    # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
        for i,j in zip(eachRoute[0::], eachRoute[1::]):
            totalDistance += calDist(dataDict['NodeCoor'][i],dataDict['NodeCoor'][j])
    return totalDistance

def evaluate(ind):
    '''评价函数，返回解码后路径的总长度，'''
    routes = decodeInd(ind) #将个体解码为路线
    totalDistance = calRoutelen(routes)
    return(totalDistance + loadPenalty(routes)) ,
#-------------------------------------------------
##交叉操作
def genChild(ind1,ind2, nTrial=5):
    '''参考《基于电动车的带时间窗的路径优化问题研究》中给出的交叉操作，生成一个子代'''
    # 在ind 中随机选择一段子路径subroute1 ，将其前置
    routes1 = decodeInd(ind1) # 将ind1解码成路径
    numSubroute1 = len(routes1) # 子路径数量
    subroute1 = routes1[np.random.randint(0,numSubroute1)]
    # 将subroute1中没有出现的顾客按照其在ind2 中的顺序排列成一个序列
    unvisited = set(ind1) - set(subroute1) # 在subroute1中没有出现访问的顾客
    unvisitedPerm = [digit for digit in ind2 if digit in unvisited] # 按照在ind2 中的顺序排列
    # 多次重复随机打断，选取适应度最好的个体
    bestRoute = None #容器
    bestFit = np.inf
    for _ in range(nTrial):
        # 将该序列随机打断为numsubroute1-1条子路径
        breakPos = [0]+random.sample(range(1,len(unvisitedPerm)),numSubroute1-2) # 产生numSubroute1-2个断点
        breakPos.sort()
        breakSubroute = []
        for i,j in zip(breakPos[0::],breakPos[1::]):
            breakSubroute.append([0]+unvisitedPerm[i:j]+[0])
        breakSubroute.append([0]+unvisitedPerm[j:]+[0])
        # 更新适应度最佳的打断方式
        # 将先前取出的subroute1 添加入打断结果，得到完整的配送方案
        breakSubroute.append(subroute1)
        # 评价生成的子路径
        routesFit = calRoutelen(breakSubroute) + loadPenalty(breakSubroute)
        if routesFit < bestFit:
            bestRoute = breakSubroute
            bestFit = routesFit
        #将得到的适应度最佳路径bestroute合并为一个染色体
        child = []
        for eachRoute in bestRoute:
            child += eachRoute[:-1]
        return child+[0]
    def crossover(ind1,ind2):
        '''交叉操作'''
        ind1[:],ind2[:] = genChild(ind1,ind2),genChild(ind2,ind1)
        return ind1,ind2
    #----------------------------------------------------
    ## 突变操作
    def opt(route,dataDict=dataDict,k=2):
        # 用2-opt 算法优化路径
        # 输入
        #route -- sequence.记录路径
        #  输出：优化后的路径optimizedroute及其路径长度
        nCities = len(route)  # 城市数
        optimizedRoute = route # 最优路径
        minDistance = calRoutelen([route]) #  最优路径长度
        for i in range(1,nCities-2):
            for j in range(i+k,nCities):
                if j-i == 1:
                    continue
                reversedRoute = route[:i]+route[i:j][::-1]+route[j:]# 翻转后的路径
                reversedRouteDist = calRoutelen([reversedRoute])
                # 如果翻转后的路径更优，则更新最优解
                if reversedRouteDist < minDistance:
                    minDistance = reversedRouteDist
                    optimizedRoute = reversedRoute
        return optimizedRoute
    def mutate(ind):
        '''用2-opt算法对各条子路径进行局部优化'''
        routes = decodeInd(ind)
        optimizedAssembly = []
        for eachRoute in routes:
            optimizedRoute = opt(eachRoute)
            optimizedAssembly.append(optimizedRoute)
            # 将路径重新组装成染色体
            child = []
            for eachRoute in optimizedAssembly:
                child += eachRoute[:-1]
                ind[:] = child+[0]
                return ind,

    #---------------------------------
    ## 注册遗传算法操作
    toolbox = base.Toolbox()
    toolbox.register('individual',tools.initIterate,creator.Individual,genInd)
    toolbox.register('population',tools.initRepeat,list,toolbox.individual)
    toolbox.register('evaluate',evaluate)
    toolbox.register('select',tools.selTournament,tournsize=2)
    toolbox.register('mate',crossover)
    toolbox.register('mutate',mutate)

    ## 生成初始族群
    toolbox.popSize = 100
    pop = toolbox.population(toolbox.popSize)

    ## 记录迭代数据
    stats= tools.Statistics(key=lambda  ind:ind.fitness.values)
    stats.register('min',np.min)
    stats.register('avg',np.mean)
    stats.register('std',np.std)
    hallOfFame = tools.HallOfFame(maxsize=1)

    ## 遗传算法参数
    toolbox.ngen = 400
    toolbox.cxpb = 0.8
    toolbox.mutpb = 0.1

    ## 遗传算法主程序
    ## 遗传算法主程序
    pop,logbook=algorithms.eaMuPlusLambda(pop,toolbox,mu=toolbox.popSize,
                                          lambda_=toolbox.popSize,cxpb=toolbox.cxpb,mutpb=toolbox.mutpb,
                                          ngen=toolbox.ngen,stats=stats,hallOfFame=hallOfFame,verbose=True)
    from pprint import pprint
    def calLoad(routes):
          loads = []
          for eachRoute in routes:
            routeload = np.sum([dataDict['Demand'][i] for i in eachRoute])
            loads.append(routeload)
          return loads
    bestInd = hallOfFame.items[0]
    distributionPlan = decodeInd(bestInd)
    bestFit = bestInd.fitness.values
    print('最佳运输计划为：')
    pprint(distributionPlan)
    print('最短运输距离为：')
    print(bestFit)
    print('各辆车上负载为：')
    print(calLoad(distributionPlan))
    # 画出迭代图
    minFit = logbook.select('min')
    avgFit = logbook.select('avg')
    plt.plot(minFit, 'b-', label='Minimum Fitness')
    plt.plot(avgFit, 'r-', label='Average Fitness')
    plt.xlabel('# Gen')
    plylabelt('Fitness')
    plt.legend(loc='best')




