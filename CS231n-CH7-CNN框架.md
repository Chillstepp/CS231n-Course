---
title: CS231n-CH7-CNN框架
date: 2021-04-17 17:47:28
index_img: /img/cs213n.png
tags: 计算机视觉CV
---

# CNN框架

![image-20210415173026774](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210415173026774.png)

## AlexNet

AlexNet结构如下：

![image-20210415173124918](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210415173124918.png)



![image-20210415173358095](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210415173358095.png)

网络结构：

![image-20210415174601793](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210415174601793.png)



第一层有96个卷积核，每个卷积核的大小是11\*11\*3,所以第一层输出的大小是: 55\*55\*96。

那么第一层的参数有：(11\*11\*3)\*96 = 35K.



第二层，即POOL1，这层在做Max pooling输出的结果大小是27\*27*96.

这一层没有参数，因为是在做Max Pooling，取最大值即可。



![image-20210415175146259](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210415175146259.png)

AlexNet的一些细节：

![image-20210415175307835](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210415175307835.png)

第一次使用ReLu损失函数，做了Normalizaiton，有很多数据增强的数据，dropout的概率是0.5，batchsize为128，梯度下降使用的是SGD Momentum,学习率1e-2开始训练，weight decay设置为5e-4，还用了模型集成，训练多个模型对他们取平均...



![image-20210416163734112](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210416163734112.png)

我们之前算第二层的大小是55\*55*96 ，而上图红色框种的深度是48，这是由于当时最好的GPU也只是GTX580，显存只有3G，所以不得不采用分开的训练方式。



## VGGNet

**这篇文章写的很好：**

一文读懂VGG网络 - Amusi的文章 - 知乎 https://zhuanlan.zhihu.com/p/41423739



![image-20210416165807394](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210416165807394.png)

VGGNet一般16-19层，它只包含3\*3的卷积(pad 1),Max Pooling是2\*2的。

- **为什么用小的卷积核？**

  **reference：**[知乎：一文读懂VGG网络](https://zhuanlan.zhihu.com/p/41423739)

  ​	VGG16相比AlexNet的一个改进是**采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）**。对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。

  ​	简单来说，在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核，这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果。

  ​	比如，3个步长为1的3x3卷积核的一层层叠加作用可看成一个大小为7的感受野（其实就表示3个3x3连续卷积相当于一个7x7卷积），其参数总量为 $3\times(9\times C^2)$ ，如果直接使用7x7卷积核，其参数总量为 $49\times C^2$ ，这里 C 指的是输入和输出的通道数。很明显，$27\times C^2\le 49\times C^2$，即**减少了参数**；而且3x3卷积核有利于更好地保持图像性质。

- **怎么使用2个3x3卷积核可以来代替5\*5卷积核**

想做到2个3\*3卷积核可以来代替5\*5卷积核,也就是对于同样一个单位的感受野，2个3\*3的卷积核也可以看到5\*5的感受野：

可以这样来做，通过两层的3\*3卷积使得第三层的一个单位可以看到第一层5*5的视野：

![image-20210416180456519](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210416180456519.png)





## GoogleNet

![image-20210416184736509](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210416184736509.png)

视频讲的有带点乱,找了些资料梳理了下。

reference：

[深度学习|经典网络 GoogLeNet（一）](https://zhuanlan.zhihu.com/p/73857137)

[深入理解GoogLeNet结构（原创）](https://zhuanlan.zhihu.com/p/32702031)

- **网络结构：**

![v2-766c3f59d3791da39ad805606d6445f6_r](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/v2-766c3f59d3791da39ad805606d6445f6_r.jpg)

- GoogleNet新提出了Inception，这是什么？

  Inception就是把多个卷积或池化操作，放在一起组装成一个网络模块，设计神经网络时以模块为单位去组装整个网络结构。模块如下图所示

![image-20210416193845211](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210416193845211.png)

​		在未使用这种方式的网络里，我们一层往往只使用一种操作，比如卷积或者池化，而且卷积操作的卷积核尺寸也是固定大小的。但是，**在实际情况下，在不同尺度的图片里，需要不同大小的卷积核，这样才能使性能最好，或者或，对于同一张图片，不同尺寸的卷积核的表现效果是不一样的，因为他们的感受野不同**。所以，**我们希望让网络自己去选择，Inception便能够满足这样的需求**，一个Inception模块中并列提供多种卷积核的操作，网络在训练的过程中通过调节参数自己去选择使用，同时，**由于网络中都需要池化操作，所以此处也把池化层并列加入网络中**。



- **为什么要提出Inception？**

  一般来说，提升网络性能最直接的办法就是增加网络深度和宽度，但一味地增加，会带来诸多问题：
  1）参数太多，如果训练数据集有限，很容易产生过拟合；
  2）网络越大、参数越多，计算复杂度越大，难以应用；
  3）网络越深，容易出现梯度弥散问题（梯度越往后穿越容易消失），难以优化模型。
  我们希望在增加网络深度和宽度的同时减少参数，为了减少参数，自然就想到将全连接变成稀疏连接。但是在实现上，全连接变成稀疏连接后实际计算量并不会有质的提升，因为大部分硬件是针对密集矩阵计算优化的，稀疏矩阵虽然数据量少，但是计算所消耗的时间却很难减少。在这种需求和形势下，Google研究人员提出了Inception的方法：**能更高效的利用计算资源，在相同的计算量下能提取到更多的特征，从而提升训练结果**。

- **实际中需要什么样的Inception？**

  我们在上面提供了一种Inception的结构，但是这个结构存在很多问题，是不能够直接使用的。首要问题就是参数太多，导致特征图厚度太大。为了解决这个问题，作者在其中加入了1X1的卷积核，改进后的Inception结构如下图

![image-20210416194145863](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210416194145863.png)

​	这样做有两个好处，首先是大大**减少了参数量**，其次，**增加的1X1卷积后面也会跟着有非线性激励，这样同时也能够提升网络的表达能力**。



- **1\*1卷积有什么用？**

  ![image-20210417121105534](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417121105534.png)

  ![image-20210417123543710](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417123543710.png)

  如果不加1\*1的卷积，那么对于上图的计算量高达8.54亿次计算。因此加上一个1\*1的卷积的原因就是降低feature map的深度，如下图：

  ![image-20210417121347102](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417121347102.png)

  因此做完1\*1卷积后，结果为：

  ![image-20210417121552106](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417121552106.png)

  ![image-20210417121724855](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417121724855.png)

  现在的操作降低到了3.58亿次操作。

  

  **知乎上的解释：**

  ![image-20210416195129813](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210416195129813.png)

  ![image-20210416194913131](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210416194913131.png)

  这里要注意：题目中所说的1\*1是对深度做操作的，1\*1卷积就是把feature map上的每个位置的c个channel做了一次重新组合,一般用来提高或者降低channel数。

  ![image-20210416194932237](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210416194932237.png)

- **多个尺寸上进行卷积再聚合的原因？**

  可以看到对输入做了4个分支，分别用不同尺寸的filter进行卷积或池化，最后再在特征维度上拼接到一起。这种全新的结构有什么好处呢？

  ![image-20210417151459105](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417151459105.png)

  ![image-20210417155307211](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417155307211.png)

## ResNet

**Reference：**

Resnet到底在解决一个什么问题呢？ - 薰风初入弦的回答 - 知乎 https://www.zhihu.com/question/64494691/answer/786270699

![image-20210417124340194](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417124340194.png)

152层...

起因是有人做了实验，发现56层不仅在训练集上效果好(当然这是应该的，人们认为这是过拟合的原因)，可是在测试集上依然比20层的网络错误率低，这就说明56层的网络并没有过拟合。

![image-20210417124603813](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417124603813.png)



![image-20210417125452353](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417125452353.png)

​	当我们堆叠一个模型时，理所当然的会认为效果会越堆越好。因为，假设一个比较浅的网络已经可以达到不错的效果，**那么即使之后堆上去的网络什么也不做，模型的效果也不会变差**。然而深度学习很难做到这一点：

![image-20210417160620669](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417160620669.png)

​	这样层层的向前传播使得开始的信息还能被完整的保留的可能性太小。

​	ResNet的出现就是为了**让模型的内部结构至少有恒等映射的能力**。以保证在堆叠网络的过程中，网络至少不会因为继续堆叠而产生退化。所以这也是ResNet敢用152层的原因。



-  **ResNet做是如何做到恒等映射的呢？**

前面提到了学习一个恒等映射很难，但是学习一个全为0的函数很简单，因此ResNet把网络设计为$H(x) = F(x) + x$,即把恒等映射作为网络的一部分。由于我们的学习目标是$F(x)$，就可以把问题转化为学习残差函数$F(x) = H(x) -x$.  

![image-20210417162541557](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210417162541557.png)

这样当$F(x)=0$（这个是易学的）时即说明这是一个恒等映射($H(x) = x$)

