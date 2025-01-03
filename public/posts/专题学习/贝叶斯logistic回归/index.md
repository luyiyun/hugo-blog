# 贝叶斯Logistic回归

# 1 Binary Logistic Regression Model

## 极大后验估计
设$X$是$n\times p$的design matrix，包含$p$个features和$n$个samples。$y$是长度为$n$的label向量，值为$0$和$1$。假设$w$是长度为$p$的参数向量，则整个模型可以表示为：

{{&lt; raw &gt;}}
$$  
P(y_i=1|x_i,w)=\sigma(w^Tx_i)=\frac{1}{1&#43;\exp(-w^Tx_i)}  
$$
{{&lt; /raw &gt;}}

假设$w$的先验分布为：

{{&lt; raw &gt;}}
$$  
w_i\sim\mathcal{N}(0,v)  
$$
{{&lt; /raw &gt;}}
则我们可以通过最小化下面的目标函数来得到$w$的MAP估计：

{{&lt; raw &gt;}}
$$  
L=-\sum_{i=1}^n{[y_i\log(\sigma(w^Tx_i))&#43;(1-y_i)\log(1-\sigma(w^Tx_i))]}&#43;\frac{1}{2v}w^Tw  
$$
{{&lt; /raw &gt;}}
注意，如果我们将$y$修改为$&#43;1$和$-1$（分别对应$1$和$0$，使用$\tilde{y}$表示），则以上的目标函数可以改写为下面的形式：

{{&lt; raw &gt;}}
$$  
L=\sum_{i=1}^n{\log(1&#43;\exp(-\tilde{y}_iw^Tx_i))}&#43;\frac{1}{2v}w^Tw  
$$
{{&lt; /raw &gt;}}
这需要用到sigmoid函数的重要性质：
&gt; [!importance] sigmoid的性质
&gt; {{&lt; raw &gt;}}
$$1-\sigma(z)=\sigma(-z)$$
{{&lt; /raw &gt;}}

当$v\to\infty$，上述MAP估计退化到MLE估计。

优化上述目标函数一般使用牛顿法，即： ^61e53c
1. 首先求出目标函数的gradient和Hessian：
    
{{&lt; raw &gt;}}
$$  
    \begin{align}
    g&amp;=-\sum_{i=1}^n{(1-\sigma(\tilde{y}_iw^Tx_i))\tilde{y}_ix_i}&#43;\frac{w}{v} \\
    &amp;=\sum_{i=1}^n{(\sigma(w^Tx_i)-y_i)x_i}&#43;\frac{w}{v}\\
    H&amp;=\sum_{i=1}^n{\sigma(y_iw^Tx_i)(1-\sigma(y_iw^Tx_i))x_ix_i^T}&#43;\frac{1}{v}I_p \\
    &amp;=\sum_{i=1}^n{\sigma(w^Tx_i)(1-\sigma(w^Tx_i))x_ix_i^T}&#43;\frac{1}{v}I_p \end{align}  
    $$ ^637fb6
{{&lt; /raw &gt;}}
2. 进行如下迭代计算直到梯度接近0：
    
{{&lt; raw &gt;}}
$$  
    w^{t&#43;1}=w-\alpha H^{-1}g  
    $$
{{&lt; /raw &gt;}}
    其中$\alpha$是步长，可以设置为$1$，但是最好通过line search methods来找到sufficient descent conditions。
3. 注意到，Hessian是negative-definite，因此目标函数是concave，一定能够得到global minimizer。
## 后验期望估计
有两种做法：
* 通过[[重要性采样#示例：贝叶斯Logistic回归|重要性采样]]的方式。
* 使用后面提及的MCMC进行采样，然后计算均值。

## 后验分布估计
基于牛顿法只能得到点估计，如果能够对后验分布进行估计，则还能得到更多的信息。

### Gibbs Sampling
首先我们考虑使用Gibbs抽样，其是通过添加auxiliary variables来实现Gibbs抽样。相似的思想也可以应用到贝叶斯Probit回归上，并且更加直观，请先了解[[贝叶斯Probit回归|贝叶斯Probit回归的auxiliary gibbs sampling]]。

基于贝叶斯Probit回归的思想，在logistic regression上的一个最直接的想法就是使用logistic distribution来替换standard normal distribution，即：

{{&lt; raw &gt;}}
$$  
\epsilon_i\sim \text{Logistic}(0, 1)  
$$
{{&lt; /raw &gt;}}
&gt; [!NOTE]
&gt; title: $\text{Logistic}$的密度函数和分布函数
&gt; collapse: closed
&gt; 
&gt; 其分布函数为就是sigmoid函数，
&gt; {{&lt; raw &gt;}}
$$L(x)=\frac{1}{1&#43;exp(-x)}$$
{{&lt; /raw &gt;}}
&gt; 其密度函数是其导数，
&gt; {{&lt; raw &gt;}}
$$f(x)=L(x)(1-L(x))=\frac{\exp(-x)}{(1&#43;\exp(-x))^2}$$
{{&lt; /raw &gt;}}

利用相同的推导思路，我们可以得到下面的Gibbs抽样过程：
&gt; [!error]
&gt; title: Logistic模型的Gibbs采样过程（错误）
&gt; 
&gt; {{&lt; raw &gt;}}
$$  
\begin{align}
z_i|w,y,x_i&amp;\sim \left\{\begin{matrix}
	\text{Logistic}(w^Tx_i,1)I(z_i&gt;0) &amp; \text{if}\ y_i=1 \\
	\text{Logistic}(w^Tx_i,1)I(z_i\le0) &amp; \text{if}\ y_i=0
	\end{matrix}\right. \\
w|y,X,z&amp;\propto \text{Logistic}(Xw, I_n)\mathcal{N}(0, v)
\end{align}
$$
{{&lt; /raw &gt;}}
&gt; 对$w$进行采样时，我们需要求解一个后验，其先验是Normal，似然是Logistic。这个后验无法显式地写出来，所以其实无法很容易地进行采样。
&gt; 
&gt; 我之前推导出现了错误，在采样$w|y,X,z$的时候，依然使用正态分布采样，得到下面的结果(预热10000，取样10000)：
&gt; ![[image_CI7NX_ZFz9.png|logistic模型的gibbs采样(横线表示真实值)]]

文献[^2]中介绍了另外一种基于auxiliary variables的gibbs采样，需要利用到Kolmogorov-Smirnov distribution（简记为KS distribution）。

KS distribution是下面的random variable所服从的分布

{{&lt; raw &gt;}}
$$
K=\sup_{t\in[0,1]}|B(t)|
$$
{{&lt; /raw &gt;}}
其中$B(t)$是[Brownian bridge](https://en.wikipedia.org/wiki/Brownian_bridge)。
![[Pasted image 20240221222611.png|KS distribution的概率密度函数]]
$K$的累计分布函数可以表示为

{{&lt; raw &gt;}}
$$
P(K\le x)=1-2\sum_{k=1}^{\infty}{(-1)^{k-1}\exp(-2k^2x^2)}=\frac{\sqrt{2\pi}}{x}\sum_{k=1}^{\infty}{\exp(\frac{-(2k-1)^2)\pi^2}{8x^2})}
$$
{{&lt; /raw &gt;}}


---

> 作者: [rongzhiwei](https://rongzhiwei.github.io/)  
> URL: http://localhost:1313/posts/%E4%B8%93%E9%A2%98%E5%AD%A6%E4%B9%A0/%E8%B4%9D%E5%8F%B6%E6%96%AFlogistic%E5%9B%9E%E5%BD%92/  

