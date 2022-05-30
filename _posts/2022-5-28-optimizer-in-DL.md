---
title: 深度学习中的优化器
tags: DL
# mathjax_autoNumber: true
---

### 背景
深度学习可以归结为一个优化问题，最小化目标函数$J(\theta)$；优化的求解过程，首先求解目标函数的梯度$\nabla J(\theta)$，然后将参数$\theta$向负梯度方向更新，$\theta_t=\theta_{t-1}-\eta\nabla J(\theta)$，$\eta$是学习率。深度学习优化器的两个核心就是梯度与学习率，前者决定了参数更新的方向后者决定了参数更新的程度。深度学习之所以采用梯度是因为，对于高维的函数其高阶导的计算复杂度大，应用到深度学习的优化中不实际。

深度学习的优化器可以分为两个大类，第一类是优化过程中，学习率不受梯度影响，全程不变或者按照一定的learning schedule随时间变化，这类包括最常见的SGD（随机梯度下降法），带Momentum的SGD，带Nesterov的SGD，这一类可以叫做SGD系列；另一类是优化过程中，学习率随着梯度自适应的改变，并尽可能消除给定的全局学习率的影响，这一类的优化器有很多，常见的有Adagrad，Adadelta，RMSprop，Adam；还有AdaMax Nadam Adamax NadamAMSgrad(ICLR 2018 best paper) Adabound，这一系列可以称为自适应学习率系列。

### SGD系列

+ GD (Gradient descent)
    
    不算SGD系列，就是每次迭代都要遍历所有样本$$\left\{x\right\}_{i=1}^n$$。
    
    GD的两个缺点：

    1. 训练速度慢：每进行一步都要调整下一步的方向，在大型数据中，**每个样本都更新一次参数，且每次迭代都要遍历所有样本**，需要很长时间进行训练和达到收敛。
    2. 容易陷入局部最优解：在有限的范围内寻找路径，当陷入相对平坦的区域，梯度接近0，参数更新缓慢。

+ BGD 批量梯度下降
    
    用所有数据进行训练。每一次参数的更新都用到了所有的训练数据，如果训练数据非常多的话，是非常耗时的。梯度公式$\nabla J(\theta) = \frac{1}{n}\Sigma_{i=1}^{n}\nabla J_i(\theta)$


+ SGD 随机梯度下降

    每次训练加载对所有样本进行随即均匀采样得到的一个样本，利用该样本的特征来对目标函数进行迭代。这样做的好处是减小了每次迭代的计算开销，更新公式：
    $$
        \theta = \theta - \eta \cdot \nabla_\theta J_i(\theta) \quad i\in\{1, 2,...,n\}
    $$
    优点：对于大型数据，计算速度快；引入随机性，避免陷入局部最优。
    
    缺点：噪音较BGD要多，权值更新方向可能出现错误，使得SGD并不是每次迭代都向着整体最优方向。

+ MBGD 小批量梯度下降

    小批量梯度下降是为了解决批梯度下降法的训练速度慢，以及随机梯度下降法的准确性问题综合得到的算法。小批量随机梯度下降的实现是在SGD的基础上，随机取batch个样本，而不是1个样本。

+ SGDM Momentum 动量

    更新公式：
    $$
    \begin{cases}
        v_t = \gamma v_{t-1}+\eta\nabla_\theta J(\theta)
        \\
        \theta = \theta - v_t 
    \end{cases}
    $$
    其中，$\gamma$代表了动量占比。动量解决了SGD的两个问题：
    1. SGD引入的噪声
    2. Hessian矩阵病态
    
    使得网络更优和更稳定的收敛，减少振荡过程。
+ 动量优化算法——NAG（Nesterov accelerated gradient）

    NAG是Momentum的变种，更新公式如下：
    $$
        \begin{cases}
            v_t = \gamma v_{t-1} + \eta\nabla_\theta J(\theta - \gamma v_{t-1})\\
            \theta = \theta - v_t
        \end{cases}
    $$
    NAG用$\theta-\gamma v_{t-1}$来近似当作参数下一步会变成的值，则在计算梯度时，不是在当前位置而是在未来的位置上。
    
    优点：在梯度更新时做一个校正，避免前进太快，同时提高灵敏度。

### 自适应系列

+ Adagrad

    通过以往的梯度来自适应地更新学习率，不同的参数$\theta_i$有不同的学习率。Adagrad对常出现的特征进行小幅度的更新，不常出现的特征进行大幅度的更新，因此适合处理稀疏数据。

    SGD对每个参数$\theta_i$在每步$t$的更新公式为
    $$
        \theta_{t+1,i} = \theta_{t,i}-\eta \cdot g_{t,i}
    $$
    Adagrad对每个参数在每步的更新公式为：
    $$
        \theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii}+\epsilon}} \cdot g_{t,i}
    $$
    其中，$G_t\in\mathbb{R}^{d\times d}$是一个对角阵，它的每个元素是到第$t$步的梯度平方和。随着参数更新，梯度平方和逐渐增大，学习率减小并最终区域零。
+ Adadelta

    Adadelta是对Adagrad的一个改进，它解决了Adagrad优化过程中学习率单调减小的问题。Adagdelta不再对过去的梯度平方进行累加，而是改用指数平均的方法计算$G_t$，Adagrad中的$G_t$替换为$E[g^2]_t$，
    $$
        E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma)g^2_t = (1-\gamma)\Sigma_{i=0}^t \gamma^i g_t^2
        \\
        \Delta\theta_t = -\frac{\eta}{\sqrt{E[g^2]_t+\epsilon}}g_t=-\frac{\eta}{RMS[g]_t}g_t
    $$
    $RMS[g]_t$对应$g_t$的均方根。但是，此时的Adadelta依然依赖于全局的学习率$\eta$，为了消除全局学习率的影响，定义新的指数平均方法，
    $$
        E[\Delta \theta^2]_t=\gamma E[\Delta \theta^2]_{t-1}+(1-\gamma)\Delta\theta^2_t
        \\
        RMS[\Delta \theta]_t = \sqrt{E[\Delta \theta^2]_t + \epsilon}
    $$
    最后得到Adadelta的更新规则为，$\Delta\theta_t=-\frac{RMS[\Delta\theta]_{t-1}}{RMS[g]_t}g_t$
+ RMSprop
    
    RMSprop就是Adadelta的第一种形式，$\gamma$通常设为0.99，$\eta$通常设为0.001。
    $$
        E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma)g^2_t = (1-\gamma)\Sigma_{i=0}^t \gamma^i g_t^2
        \\
        \Delta\theta_t = -\frac{\eta}{\sqrt{E[g^2]_t+\epsilon}}g_t=-\frac{\eta}{RMS[g]_t}g_t
    $$
+ Adam
    
    在SGDM的基础上，引入自适应学习率：
    $$
        m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
        \\
        v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
    $$
    $m_t,v_t$分别对应梯度的一阶和二阶矩估计，是有偏估计，将其校正为无偏估计：
    $$
        \hat{m_t} = \frac{m_t}{1-\beta_1^t}\\
        \hat{v_t} = \frac{v_t}{1-\beta_2^t}
    $$
    最终得到Adam的更新规则：$\theta_{t+1} = \theta_t-\frac{\eta}{\sqrt{\hat{v_t}+\epsilon}}\hat{m_t}$

### 自适应 v.s. SGD

> 自适应算法通常会得到比SGD算法更差的结果，尽管自适应算法在训练时会表现得比较好，在使用自适应优化算法时需要慎重考虑。（CVPR的paper全部使用SGD，而不是Adam）

关于这个问题，[这篇博客](https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008)有着详细的讨论，其大概内容是：
+ 一直以来，众多研究发现SGD有着比Adam更好的泛化性
    + 如果通过训练和测试误差之间的差异衡量算法的稳定性，SGD对于连续凸优化问题在理论上是稳定的
    + 当优化问题有许多局部极小值，不同算法的解完全不同，非自适应算法（包括SGD和momentum）在二分类最小二乘loss任务中可以收敛到范数最小的解，而自适应算法会发散
    + 虽然Adam看上去不怎么需要调参，但是通过调节它的初始学习率和decay scheme可以显著提升其性能
+ 真的是这样吗？最近的一篇paper发现**超参数**是制约自适应算法泛化的原因。增大超参数搜索空间，自适应算法可以有更好的泛化性，不过最优超参数随数据集的不同有很大差异。从另一个角度想，自适应算法是更一般的算法，SGD/momentum是其特例，因此给自适应算法足够大的超参数调整空间，它就不会比它的特例差到哪里去。

**总结一下，我们可以说fine-tune过的Adam算法是要比SGD好的，然而当我们使用默认超参数的时候，Adam和SGD之间存在性能差异。**

### 参考
[1] [深度学习之——优化器](https://www.jianshu.com/p/7149f519c5c3)

[2] [深度学习优化器总结](https://zhuanlan.zhihu.com/p/58236906)

[3] [A 2021 Guide to improving CNNs-Optimizers: Adam vs SGD](https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008)