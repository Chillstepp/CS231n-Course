---
title: CS231n-CH4-卷积神经网络
date: 2021-04-09 18:35:24
index_img: /img/cs213n.png
tags: 计算机视觉CV
---

# 卷积神经网络

## 卷积

全连接层：

![image-20210409162420380](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409162420380.png)

卷积层(convolution Layer):

![image-20210409162546045](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409162546045.png)



卷积网络就是一系列的卷积层的叠加，并加上各种各样的激活函数。

![image-20210409163643703](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409163643703.png)



随着层数的增加，所包含的信息越来越多。

![image-20210409163844238](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409163844238.png)



![image-20210409164331727](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409164331727.png)





卷积计算过程：

我们可以一步一步的来走，那么得到的就是 5*5的结果

![image-20210409164505656](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409164505656.png)

但是步长调整到2时：

![image-20210409164739985](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409164739985.png)

那么得到的就是一个3*3的结果。



步长为3时：(7-3)/3+1 = 2.33 所以这样会导致不平衡的结果，所以步长为3不可取。

![image-20210409164831571](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409164831571.png)



我们也可以在周围添上0，这样就可以在边缘上做卷积：

![image-20210409165253842](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409165253842.png)



例如：一个32\*32\*3的输入，用10个5\*5\*3的卷积核去做卷积，边缘填充为2,步长为1。那么最后得到的就是32\*32\*10的输出，首先是32怎么得出来的呢：

(32+2*2-5)/1 + 1 = 32。

由于有10个卷积核，那么就是10层。

![image-20210409170501645](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409170501645.png)

那么这个例子中包含多少参数呢：

![image-20210409171108179](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409171108179.png)

首先是每kernel的大小是5\*5\*3 = 75,还有一个bias项，所以一共是76，由于有10层，那么就是760个参数。



1*1的卷积也有意义：

![image-20210409171525184](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409171525184.png)

可以对深度进行操作。



## 池化

![image-20210409172416207](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409172416207.png)

感受野(neuron view)：一个神经元所看到的视野，如上图蓝色面上的一个神经元是由卷积核和input做卷积得到的，他们操作的范围是5\*5的范围。



![image-20210409173042186](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409173042186.png)

池化层(pooling layer)做的是降采样(downsampling)：

![image-20210409173144211](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409173144211.png)

我们一般不会做深度方向的池化，只对平面做。所以我们池化后的深度是不会变的。

一种常用的池化方法是Max Pooling。

Max Pooling：

![image-20210409173511648](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210409173511648.png)

除了pooling可以降采样，我们调整卷积时的步长也可以做到降采样。



