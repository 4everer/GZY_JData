# For JData compeition

[competition page](http://www.datafountain.cn/#/competitions/247/intro)

aims

```
任务描述：
参赛者需要使用京东多个品类下商品的历史销售数据，构建算法模型，预测用户在未来5天内，对某个目标品类下商品的购买意向。对于训练集中出现的每一个用户，参赛者的模型需要预测该用户在未来5天内是否购买目标品类下的商品以及所购买商品的SKU_ID。评测算法将针对参赛者提交的预测结果，计算加权得分。
```

To download data: 
- [link1](http://122.5.18.194:28080/jdata/JData.7z) or 
- [link2](http://221.0.111.140:28080/jdata/JData.7z) or 
- [link3](http://223.99.62.203:28080/jdata/JData.7z)

ipython notebook for data exploration and data transformation

package used: pandas, numpy, matplotlib, sklearn

## Strategy

First predict customers that will make a purchase.
Then use their history to predict products that they will get.

## Algorithm: Expectation-Maximization based max likelihood estimation, using mini-batch

### Model
Users can be classified into $m$ groups, with unique parameters for each group.

User behavior follows poisson distribution, for behavior type $i$, the poisson intensity for group $j$ is $\lambda_ji$.
P(behavior|user in group j) is the corresponding poisson probability.

Assuming that the prior probability for a user in any group is equal, then Bayesian probability for user in group j is given as,
$$ P(user in group j|behavior) = \frac{P(behavior|user in group j)}{\sum_k {P(behavior|user in group k)}} $$

Assuming that the effect of an action generates an exponential impulse of $e^{-\lambda_ji}$, then the probability for purchase at a time point is
$$ P(purchase) = \sum_i {P(purchase|behavior i) * \sum {behavior * impulse}} $$


### Algorithm
The algorithm alternatively updates P(purchase|behavior i) or $\lamda_ji$. The group for users in a mini-batch is determined by probability calculated from $\lambda$.

Update is performed with momentum $\mu$ to reduce the effect of noise.

Test is performed at every 100 iterations, to evaulate error.




