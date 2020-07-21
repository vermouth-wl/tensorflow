import tensorflow as tf

# 向量相加
with tf.compat.v1.Session() as sess:

    # 向量赋值

    a = tf.constant([1.1, 2.2], name='a')
    b = tf.constant([3.3, 4.4], name='b')
    result = a + b

    # 调用run()
    print(sess.run(result))