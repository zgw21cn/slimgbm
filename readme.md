SlimGBM基于wepe的[tgboost](https://github.com/wepe/tgboost),参考[lightGBM](https://github.com/Microsoft/LightGBM)，作了如下修改：
1. 基于histogram计算最佳分割点；
2. Leaf-wise的树增长方式；

原有的tgboost支持如下特征：
- 内置有Square error losst和Logistic loss
- 可自定义损失函数，使用'autograd'计算grad和hess
- 多线程寻找最佳分割点
- 计算特征重要性
- 处理缺失值（slimgbm为方便去掉了此功能）
- 正则化
- 随机选取特征和样本
- 支持对样本定义权重函数

