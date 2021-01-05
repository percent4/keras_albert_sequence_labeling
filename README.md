本项目采用Keras和ALBERT实现序列标注，其中对ALBERT进行微调。

### 维护者

- jclian91

### 数据集

1. 人民日报命名实体识别数据集（example.train 28046条数据和example.test 4636条数据），共3种标签：地点（LOC）, 人名（PER）, 组织机构（ORG）
2. 时间识别数据集（time.train 1700条数据和time.test 300条数据），共1种标签：TIME
3. CLUENER细粒度实体识别数据集（cluener.train 10748条数据和cluener.test 1343条数据），共10种标签：地址（address），书名（book），公司（company），游戏（game），政府（goverment），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

### 模型结构

albert_tiny, example数据集

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
model_2 (Model)                 multiple             4077496     input_1[0][0]                    
                                                                 input_2[0][0]                    
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, None, 128)    193024      model_2[1][0]                    
__________________________________________________________________________________________________
crf_1 (CRF)                     (None, None, 7)      966         bidirectional_1[0][0]            
==================================================================================================
Total params: 4,271,486
Trainable params: 4,271,486
Non-trainable params: 0

```

### 模型效果

- 人民日报命名实体识别数据集

1.1 albert-tiny

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
           precision    recall  f1-score   support

      LOC     0.8266    0.8171    0.8218      3658
      ORG     0.7289    0.7863    0.7565      2185
      PER     0.8865    0.8712    0.8788      1864

micro avg     0.8111    0.8215    0.8163      7707
macro avg     0.8134    0.8215    0.8171      7707
```

1.2 albert-base

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
           precision    recall  f1-score   support

      LOC     0.9032    0.8671    0.8848      3658
      PER     0.9270    0.9067    0.9167      1864
      ORG     0.8445    0.8549    0.8497      2185

micro avg     0.8917    0.8732    0.8824      7707
macro avg     0.8923    0.8732    0.8826      7707
```

- 时间识别数据集

2.1 albert-tiny

模型参数：MAX_SEQ_LEN=256, BATCH_SIZE=8, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
           precision    recall  f1-score   support

     TIME     0.7924    0.8481    0.8193       441

micro avg     0.7924    0.8481    0.8193       441
macro avg     0.7924    0.8481    0.8193       441
```

2.2 albert-base

模型参数：MAX_SEQ_LEN=256, BATCH_SIZE=8, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
           precision    recall  f1-score   support

     TIME     0.8136    0.8413    0.8272       441

micro avg     0.8136    0.8413    0.8272       441
macro avg     0.8136    0.8413    0.8272       441
```

- CLUENER细粒度实体识别数据集

3.1 albert-tiny

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

     company     0.5745    0.6639    0.6160       366
organization     0.5677    0.6337    0.5989       344
        game     0.6616    0.7561    0.7057       287
    position     0.6478    0.7012    0.6734       425
  government     0.6237    0.7336    0.6742       244
        name     0.6520    0.7894    0.7141       451
       movie     0.6164    0.6533    0.6343       150
       scene     0.5166    0.5477    0.5317       199
        book     0.6140    0.6908    0.6502       152
     address     0.4071    0.4698    0.4362       364

   micro avg     0.5884    0.6687    0.6260      2982
   macro avg     0.5881    0.6687    0.6255      2982
```

3.2 albert-base

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
              precision    recall  f1-score   support

        name     0.8419    0.8381    0.8400       451
     company     0.7161    0.7650    0.7398       366
    position     0.7205    0.7459    0.7329       425
     address     0.5473    0.5879    0.5669       364
        game     0.7033    0.8258    0.7596       287
        book     0.7931    0.7566    0.7744       152
       scene     0.6243    0.5930    0.6082       199
organization     0.6711    0.7297    0.6992       344
       movie     0.7051    0.7333    0.7190       150
  government     0.7567    0.8156    0.7850       244

   micro avg     0.7078    0.7441    0.7255      2982
   macro avg     0.7093    0.7441    0.7257      2982
```

### 模型预测示例

1.1 albert_tiny

- 人民日报命名实体识别数据集

运行model_predict.py，对新文本进行预测，结果如下：

```
{'entities': [{'end': 50, 'start': 48, 'type': 'LOC', 'word': '英国'},
              {'end': 72, 'start': 69, 'type': 'PER', 'word': '卡梅伦'},
              {'end': 78, 'start': 73, 'type': 'PER', 'word': '特雷莎·梅'},
              {'end': 102, 'start': 95, 'type': 'PER', 'word': '鲍里斯·约翰逊'}],
 'string': '当2016年6月24日凌晨，“脱欧”公投的最后一张选票计算完毕，占投票总数52%的支持选票最终让英国开始了一段长达4年的“脱欧”进程，其间卡梅伦、特雷莎·梅相继离任，“脱欧”最终在第三位首相鲍里斯·约翰逊任内完成。'}
```

```
{'entities': [{'end': 6, 'start': 0, 'type': 'ORG', 'word': '台湾“立法院'},
              {'end': 30, 'start': 29, 'type': 'LOC', 'word': '台'},
              {'end': 37, 'start': 35, 'type': 'PER', 'word': '蔡英'},
              {'end': 66, 'start': 64, 'type': 'LOC', 'word': '台湾'}],
 'string': '台湾“立法院”“莱猪（含莱克多巴胺的猪肉）”表决大战落幕，台当局领导人蔡英文24日晚在脸书发文宣称，“开放市场的决定，将会是未来台湾国际经贸走向世界的关键决定”。'}
```

```
{'entities': [{'end': 9, 'start': 7, 'type': 'LOC', 'word': '印度'},
              {'end': 14, 'start': 12, 'type': 'LOC', 'word': '南海'},
              {'end': 27, 'start': 25, 'type': 'LOC', 'word': '印度'},
              {'end': 30, 'start': 29, 'type': 'LOC', 'word': '南'},
              {'end': 40, 'start': 39, 'type': 'PER', 'word': '峰'},
              {'end': 45, 'start': 43, 'type': 'LOC', 'word': '印度'},
              {'end': 49, 'start': 47, 'type': 'PER', 'word': '莫迪'},
              {'end': 53, 'start': 51, 'type': 'LOC', 'word': '南海'},
              {'end': 90, 'start': 88, 'type': 'LOC', 'word': '南海'}],
 'string': '最近一段时间，印度政府在南海问题上接连发声。在近期印度、越南两国举行的线上总理峰会上，印度总理莫迪声称南海行为准则“不应损害该地区其他国家或第三方的利益”，两国总理还强调了所谓南海“航行自由”的重要性。'}
```


- 时间识别数据集

运行model_predict.py，对新文本进行预测，结果如下：

```
{'entities': [{'end': 8, 'start': 0, 'type': 'TIME', 'word': '去年11月30日'}],
 'string': '去年11月30日，李先生来到茶店子东街一家银行取钱，准备购买家具。输入密码后，'}
```

```
{'entities': [{'end': 19, 'start': 10, 'type': 'TIME', 'word': '上世纪80年代之前'},
              {'end': 24, 'start': 20, 'type': 'TIME', 'word': '去年9月'},
              {'end': 47, 'start': 45, 'type': 'TIME', 'word': '3年'}],
 'string': '苏北大量农村住房建于上世纪80年代之前。去年9月，江苏省决定全面改善苏北农民住房条件，计划3年内改善30万户，作为决胜全面建成小康社会补短板的重要举措。'}
```

```
{'entities': [{'end': 8, 'start': 6, 'type': 'TIME', 'word': '两天'},
              {'end': 23, 'start': 21, 'type': 'TIME', 'word': '昨天'},
              {'end': 61, 'start': 56, 'type': 'TIME', 'word': '8月10日'},
              {'end': 69, 'start': 64, 'type': 'TIME', 'word': '2016年'}],
 'string': '经过工作人员两天的反复验证、严密测算，记者昨天从上海中心大厦得到确认：被誉为上海中心大厦“定楼神器”的阻尼器，在8月10日出现自2016年正式启用以来的最大摆幅。'}
```


- CLUENER细粒度实体识别数据集

运行model_predict.py，对新文本进行预测，结果如下：

```
{'entities': [{'end': 19, 'start': 14, 'type': 'address', 'word': '茶店子东街'}],
 'string': '去年11月30日，李先生来到茶店子东街一家银行取钱，准备购买家具。输入密码后，'}
```

```
{'entities': [{'end': 14, 'start': 11, 'type': 'address', 'word': '丹棱县'},
              {'end': 44, 'start': 41, 'type': 'name', 'word': '胡文和'}],
 'string': '四川敦煌学”。近年来，丹棱县等地一些不知名的石窟迎来了海内外的游客，他们随身携带着胡文和的著作。'}
```

```
{'entities': [{'end': 3, 'start': 0, 'type': 'name', 'word': '罗伯茨'},
              {'end': 10, 'start': 4, 'type': 'movie', 'word': '《逃跑新娘》'},
              {'end': 23, 'start': 16, 'type': 'movie', 'word': '《理发师佐翰》'},
              {'end': 38, 'start': 32, 'type': 'name', 'word': '亚当·桑德勒'}],
 'string': '罗伯茨的《逃跑新娘》不相伯仲；而《理发师佐翰》让近年来顺风顺水的亚当·桑德勒首尝冲过1亿＄'}
```

2.1 albert_base_zh_additional_36k_steps

3.1 albert_xlarge_zh_183k


### 代码说明

0. 将ALBERT中文预训练模型放在对应的文件夹下
1. 运行load_data.py，生成类别标签，注意O标签为0;
2. 所需Python第三方模块参考requirements.txt文档
3. 自己需要分类的数据按照data/example.train和data/example.test的格式准备好
4. 调整模型参数，运行model_train.py进行模型训练
5. 运行model_evaluate.py进行模型评估
6. 运行model_predict.py对新文本进行预测