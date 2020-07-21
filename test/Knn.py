from numpy import *
import operator


def createDataSet():
    
    """ 创建数据集和标签 """
    
    # 创建数据集, array()函数是numpy Ndarray对象
    # 数组中第一个元素是亲吻镜头，第二个元素是打斗镜头 
#     group = array([[100, 50], [900, 1500], [555, 111], [1, 10]])
    
#     # 创建标签
#     label = ['爱情片', '动作片', '爱情片', '动作片']

    group = array([[100, 90], [10, 90], [90, 90], [100, 30]])
    label = ['好瓜', '烂瓜', '好瓜', '烂瓜']
    
    return group, label


def classify0(inX, dataSet, label, k):
    
    """ 
        分类函数 
        @param: inX:输入向量，测试样本
        dataSet: 训练样本，创建的数据集
        label:标签，训练样本对应的标签
        k: 最近距离的个数
        
    """
    # shape是返回各个维度维数，类型为tuple，shape[0]返回最外围维数,shape[1]返回次外围维数,数字不断增大，维数由外到内，这儿返回数组行数
    dataSetSize = dataSet.shape[0]
    
    # dataSetSize有多行，需要将inX复制成跟训练目标向量的行数设置成一致，tile()函数可以复制inX向量
    # (dataSetSize, 1)代表复制dataSize行，如果不加上1就是复制成一行里面有dataSetSize个inX，加上就是复制4行inX
    # 最后减训练目标向量就是求输入向量和目标向量的差值
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet 
    
    # 计算差值的平方
    sqDiffMat = diffMat ** 2
    
    # 对差值平方后求和,axis=1是行相加，最后加出来的结果应是有dataSetSize个元素，axis=0是列相加
    sqDistances = sqDiffMat.sum(axis=1)
    
    # 对和进行开方
    # 此处计算距离公式为欧式距离
    sqPrescribe = sqDistances ** 0.5
    
    # 按照计算出来的距离从小到大进行排序,输出下标
    sortedDistIndices = sqPrescribe.argsort()

    classCount = {}
    
    for i in range(k):
        
        # 分别取出前k个元素的类别
        voteIlabel = label[sortedDistIndices[i]]
        
        # 存入当前label以及对应的类别值，d.get(k, v)意思是如果k在d中，则返回d[k]，否则返回v
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        
        # 对类别字典进行逆排序，级别数目多的往前放，operator.itemgetter(1)代表和元素的第一个域进行比较
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        
    return sortedClassCount[0][0]


if __name__ == '__main__':
    
    # 创建数据集
    group, label = createDataSet()

    # 输入测试样本
    test = [100, 100]
    
    # 调用分类器
    result = classify0(test, group, label, 4)
    
    # 输出结果
    print('这个西瓜是： ', result)

