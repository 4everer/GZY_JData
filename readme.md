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


