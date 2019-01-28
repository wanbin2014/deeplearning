# deeplearning
### embedding的做法
1、使用FM为每个单独特征预训练生成embedding，可以利用3层网络来生成，第一层：n*embedding_size 第二层：加入先验知识后的两两组合的内积 第三层:LR层。
#### tf.nn.embedding_lookup_sparse
tf.nn.embedding_lookup_sparse 和tf.sparse_mamtul的区别是 tf.nn.embedding_lookup_sparse可以执行梯度，并且可以给每个id带权重，由于我们需要更新embedding更新，
所以需要在预训练embedding过程中，需要使用tf.nn.embedding_lookup_sparse。
