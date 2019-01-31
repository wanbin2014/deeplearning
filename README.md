# TensorFlow的神经网络构造
### embedding的做法
1、使用FM为n个特征单独特征预训练生成embedding，可以利用3层网络来生成，第一层：n*embedding_size 第二层：加入先验知识后的两两组合的内积 第三层:LR层。

##### tf.nn.embedding_lookup_sparse
tf.nn.embedding_lookup_sparse 和tf.sparse_mamtul都是根据输入的one-hot编码，得到相应的embedding向量，区别是 可以给每个id带权重，由于我们需要更新embedding更新，所以需要在预训练embedding过程中，需要使用tf.nn.embedding_lookup_sparse。
#### 使用embeding向量
在fnn模型结构里，预加载之前预训练生成的embedding向量，在获取embedding的时候，使用tf.sparse_matmul获取（目的是让embedding向量不更新），需要保证各特征的embedding的维度是一致，然后把多个特征的embedding向量concat起来，后面开始叠加网络结构。

### conv2d的构造方法 tf.nn.conv2d
选择合适的filter的参数的shape，[embed_size,filter_size(横跨字段的个数),in_channel, out_channel]。因为在embedding里不可分割，所以第一个字段大小是embedding_size的大小。 conv2d的内部的操作如下：
···
input[batch,input_height,input_weight,in_channel] input_height是embeding_size，input_weight是字段数量
filter[filter_height,filter_weight,in_channel,out_channel]
···

 把filter转成2维的,变成filter[filter_height*filter_weight*in_channel,out_channel] 
 从input中取一小块矩阵和filter做矩阵乘法,相当于[batch,out_height,out_weight, filter_height*filter_weight*in_channel] * [filter_height*filter_weight*in_channel,out_channel]
 最后output的shape[batch,out_height,out_weight,out_channel]

当padding的方法是same,stride=[1,1,1,1]时：
out_height = input_height 
out_weight = input_weight


### max-pooling的做法
 使用tf.nn.top_k，获取最小维度的k个最大值。因为我们要取的是字段之间组合中，最有效的组合，所以需要把conv2d的输出做一次transpose(0,1,3,2)，等做完pooling后，再转回去





