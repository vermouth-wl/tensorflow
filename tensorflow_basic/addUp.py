# 使用tensorflow实现1+1
import tensorflow as tf


# 在会话里执行操作
with tf.compat.v1.Session() as sess:

    # 构建图，定义两个常量和加法操作
    v1 = tf.compat.v1.constant(1, name='value1')

    # 输出v1操作的代码
    print(v1)

    v2 = tf.compat.v1.constant(1, name='value2')

    # 输出操作v2的代码
    print(v2)

    add_op = tf.compat.v1.add(v1, v2, name='add_op_name')

    # 执行运算,将TensorFlow的计算结果赋值给python变量
    result = sess.run(add_op)
    print('1 + 1 = {}'.format(result))


    ### 获取图中的所有运算操作
    graph = tf.compat.v1.get_default_graph()

    # 获取图中的所有操作
    operations = graph.get_operations()
    print('操作数: {}'.format(len(operations)))
    print('操作: ')

    for op in operations:
        print(op)
