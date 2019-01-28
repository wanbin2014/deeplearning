# deeplearning
### embedding的做法
1、使用FM为每个单独特征预训练生成embedding，可以利用3层网络来生成，第一层：n*embedding_size 第二层：加入先验知识后的两两组合的内积 第三层:LR层。
#### 预训练
##### tf.nn.embedding_lookup_sparse
tf.nn.embedding_lookup_sparse 和tf.sparse_mamtul都是根据输入的one-hot编码，得到相应的embedding向量，区别是 tf.nn.embedding_lookup_sparse可以执行梯度，并且可以给每个id带权重，由于我们需要更新embedding更新，所以需要在预训练embedding过程中，需要使用tf.nn.embedding_lookup_sparse。
#### 使用embeding向量
在fnn模型结构里，预加载之前预训练生成的embedding向量，在获取embedding的时候，使用tf.sparse_matmul获取（目的是让embedding向量不更新），需要保证各特征的embedding的维度是一致，然后把多个特征的embedding向量concat起来，后面开始叠加网络结构。

