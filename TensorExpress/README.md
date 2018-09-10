# TensorExpress
Tensors and neural networks framework with back/front-end separated.

主要特征包括：

* 1.前后端分离设计
>* 前端脚本语言自由选择，对接标准的JSON RPC，默认提供c++/Javascript两个语言的前端实现
>* 训练和部署环境一致，都是通过RPC调用
>* 框架仅仅只是一个计算图引擎，将更多深度学习框架内容外置到前端语言环境

* 2.面向分布式的设计
>* 内置Paramter Server 从设计阶段就考虑分布式训练
>* 简化的分布式训练部署

* 3.现代框架特征
>* 动态/静态计算图同时支持
>* 支持标准交换格式onnx输出
>* 支持计算图编译优化
>* 内置MKL/cuDNN加速支持