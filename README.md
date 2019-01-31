# TensorFlow
### 关于输入处理的流程

#### np.array
数组，用于生成matrix对象
#### scipy.sparse.coo_matrix
稀疏矩阵，通常有data,row index,column index 三个一维数组作为参数生成，coo_matrix(data,(row,column))，把用户的输入的数据转成该对象
常用的方法有：
shape 
利用索引的列表可以重新生成一个新的稀疏矩阵


#### scipy.sparse.csr_matrix
 Compressed Sparse Row format ，可以有coo_matrix.tocsr()取得，通常我们总是吧coo_matrix转成它来节省空间。
 可以[] 获取里面的元素
 可以支持类似于[:,2:10]这样的slice操作
 
 #### SparseTensorValue
 当输入是稀疏矩阵的时候，需要把最终数据转换成SparseTensorValue的格式，并在输入的placeholder设置成sparse_placeholder，这样就可以把矩阵传给tensorflow的网络。
 注意该对象的indices参数是[[x1,y1],[x2,y2]]的样子，要获取这个数据，需要把crs_matrix转成tocoo()格式后，取row和col属性，再transpose。如：
 ```
 coo = crs_matrix.tocoo()
 indices = np.vstack((coo.row,coo.col)).transpose()
 ```

### 预训练生成embedding的做法
* 使用FM为n个特征单独特征预训练生成embedding，可以利用3层网络来生成，第一层：n*embedding_size 第二层：加入先验知识后的两两组合的内积 第三层:LR层。

##### tf.nn.embedding_lookup_sparse
tf.nn.embedding_lookup_sparse 和tf.sparse_tensor_dense_matmul 都是通过one-hot编码，得到相应的embedding向量，区别是 embedding_lookup_sparse可以给每个id带权重，embedding_lookup_sparse比sparse_tensor_dense_matmul更强大些。
#### 使用embeding向量
* 在fnn模型结构里，预加载之前预训练生成的embedding向量，在获取embedding的时候，使用tf.sparse_tensor_dense_matmul，需要保证各特征的embedding的维度是一致（貌似不一致也没太大关系），然后把多个特征的embedding向量concat起来，后面开始叠加网络结构。
* 如果没有预加载，使用随机初始化的参数，可以完成，貌似这种方法更简洁。个人觉得区别在于速度。



### conv2d的构造方法 tf.nn.conv2d
选择合适的filter的参数的shape，[embed_size,filter_size(横跨字段的个数),in_channel, out_channel]。因为在embedding里不可分割，所以第一个字段大小是embedding_size的大小。 conv2d的内部的操作如下：
```
input[batch,input_height,input_weight,in_channel] input_height是embeding_size，input_weight是字段数量
filter[filter_height,filter_weight,in_channel,out_channel]
```

* 把filter转成2维的,变成filter[filter_height*filter_weight*in_channel,out_channel] 
* 从input中取一小块矩阵和filter做矩阵乘法,相当于[batch,out_height,out_weight, filter_height*filter_weight*in_channel] * [filter_height*filter_weight*in_channel,out_channel]
* 最后output的shape[batch,out_height,out_weight,out_channel]

* 当padding的方法是same,stride=[1,1,1,1]时：
```
out_height = input_height 
out_weight = input_weight
```

### max-pooling的做法
* 使用tf.nn.top_k，获取最小维度的k个最大值。因为我们要取的是字段之间组合中，最有效的组合，所以需要把conv2d的输出做一次transpose(0,1,3,2)，等做完pooling后，再转回去





