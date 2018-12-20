# YOLOv2解説
## FCN(Fully Convolutional Networks)特征图提取
通常のCNNでは、最終層に全結合層を入れてsoftmax関数などにかけて、画像のclassificationを行うが、FCNでは、最初から最後まで全てがconvolution層で構築され、特徴マップの精確な位置情報を保持したまま最終層まで伝播される。Semantic Segmentationのタスクでよく用いられるモデルですが、YOLOv2でもこれを導入した事で大幅な精度向上を実現。
在普通CNN中，所有绑定层都放在最后一层，图像分类由softmax函数等执行。在FCN中，从开始到结束的所有内容都由卷积层构成，并保存特征映射的准确位置信息。 并传播到最后一层。 虽然它是语义分割任务中经常使用的模型，但YOLOv 2也通过引入它实现了准确性的显着提高。

![fcn](data/fcn_short.gif)



## 预测每个网格的边界框和置信度
YOLOv2では、入力画像の大きさによって最終層の出力がn x nの特徴マップになるが、これは入力画像をn x nの大きさに分割した時のそれぞれのgridに対応する。各gridは、複数のanchorと呼ばれる一定のアスペクト比のbounding boxを持ち、YOLOv2では各anchorの中心座標(x, y)及び、幅と高さのスケール(w, h)を予測する。更に、各anchor boxはconfidenceと呼ばれるパラメータを持っていて、これはbox内に物体が存在する確率を表す。
在YOLOv2中，最终层的输出变为n×n的特征映射，这取决于输入图像的大小，当输入图像被分成n×n个大小时，其对应于每个网格。 每个网格具有称为多个anchor的特定纵横比的边界框，并且YOLOv2预测每个锚的中心坐标（x，y）和宽度和高度的比例（w，h）。 此外，每个锚框都有一个名为confidence的参数，它表示对象存在于框中的概率。

この図では各anchor boxの線の太さがconfidenceの高さを表し、最も太線の縦長いanchor boxは、そこに高確率でなにかしらの物体が存在する事を意味している。
在该图中，每个锚盒的线的粗细表示置信度，最粗线的最长锚盒表示存在某种高概率的物体。


![bbox_conf](data/bbox_confidence_pred.gif)

同様なanchor boxの予測が全てのgridで行い、そのconfidenceが一定以上(例えば50%)のboxのみをネットワークの出力に用いる。
对所有网格执行类似的锚箱预测，并且仅将置信度等于或大于特定值（例如，50％）的箱子用于网络的输出。

![bbox_conf_each_grid](data/prediction_process.gif)


## 预测每个边界框的条件概率
各anchor boxは、そこにものが存在するかのconfidenceに加え、もしものが存在するとすれば、それはなんであるかのconditional probability(条件付き確率)も予測するようにしています。
この図では各boxの色がそれぞれのクラスラベルに対応していて、赤色で囲まれているところがhuman、すなわち人間を予測している箇所となります。
除了对其中的内容的置信度之外，每个锚箱都对条件概率（条件概率）进行预测，如果它存在的话。
在该图中，每个盒子的颜色对应于每个类别标签，被红色包围的部分是人类，即预测人类的部分。

<img src="data/conditional_prob.png" width=640px>




## YOLOv2的误差计算
以上のanchor box、confidence、及び条件付き確率の予測を、YOLOv2では１つのLoss functionに統合しています。もちろん元々別問題だったタスクを1つの誤差関数として逆伝播するので、それぞれの誤差の重みは非常にデリケートですが、学習係数をうまく調整することでend-to-endの学習を可能にしています。
上述锚框，置信度和条件概率的预测被集成在YOLOv2中的一个Loss函数中。 当然，由于最初不同的任务被反向传播为一个错误函数，每个错误的权重非常微妙，但通过很好地调整学习系数可以实现端到端的学习。

<img src="data/regression.png" width=640px>

これを分割した状態でも各々訓練可能。
即使这样划分，每个人都可以接受训练。

### 条件概率的误差计算
<img src="data/conditional_prob_loss.png" width=400px>

### anchor boxの誤差計算
<img src="data/anchor_loss.png" width=900px>

### confidenceの誤差計算
<img src="data/confidence_loss.png" width=600px>


## dimension cluster
訓練データからk-means法で『最もよく使われるanchor box』を予め決定する手法。YOLOv2ではデフォルトでk=5。これを導入する事で約3〜5%の精度向上を実現。
一种从训练数据中通过k均值方法初步确定“最常用的锚箱”的方法。 在YOLOv 2中，默认情况下k = 5。 通过引入这一点，实现了约3％至5％的精度提高。


## 使用暗网进行特征提取19
物体検出のニューラルネットでは、特徴抽出器としてVGG16をベースに事前学習を行うのが一般的だが、YOLOv2ではdarknet19という独自のclassificationモデルを用いる。VGG16とほぼ同等の精度を維持しつつ、計算量を抑えてVGG16の4倍以上高速に動作する。
在对象检测神经网络中，通常基于VGG 16作为特征提取器执行初步学习，但是YOLOv 2使用称为暗网19的唯一分类模型。 在保持与VGG 16几乎相同的精度的同时，通过少量计算，它的运行速度至少比VGG 16快四倍。

- VGGと同じくカーネルを3x3とし、pooling層の後でchannel数を2倍にする。
- GoogLeNetと同様に3x3の層の間に1x1の層を入れて次元削減を行う。
- 最後の全結合層を取っ払い、代わりにFCN構造を採用。
- Batch Normalizationを全てのConv層に入れる(betaを使わない)。
- Conv層ではBiasを使わない。Batch Normの後にBias層を入れる(channel単位でBias共有)。
- softmax関数の後、cross entropyの代わりにsum of squared errorを使う。 
- activation関数にreluの代わりにleaky reluを使う。(slope=0.1)
- learning rateについて、初期値0.1で、4乗のPolynomial decay方式で減衰させる。
- NINと同様Global Average pooling層を採用し、入力解像度へ依存しない構造。
 - 与VGG一样，将内核设置为3×3，并在池化层之后将通道编号加倍。
 - 与GoogLeNet类似，在3x3层之间插入1x1层以减小尺寸。
 - 删除最后一个总绑定层，改为采用FCN结构。
 - 在所有Conv层中放置批量标准化（不要使用beta）。
 - 不要在Conv层中使用Bias。 批量标准后放置偏置层（基于通道的偏差共享）。
 - 在softmax函数之后，使用平方误差之和而不是交叉熵。
 - 使用泄漏的relu而不是relu激活功能。 （斜率= 0.1）
 - 对于学习速率，使用初始值为0.1和第四次幂的多项式衰减方法进行衰减。
 - 采用全局平均池化层和NIN的结构，不依赖于输入分辨率。


## 结合高分辨率和低分辨率的特征图

CNNでは層が深くなるに連れ特徴マップの解像度が落ちるので、より精確なanchor boxの座標予測をするために、YOLOv2では、解像度の高い層からサンプリングした特徴マップを解像度の低い層とチャンネル間で結合している。通常、特徴マップのサイズが合わないとチャンネル間で結合できないが、YOLOv2ではreorganizationという独自の手法を使って、1枚の高解像度の特徴マップを複数の低解像度の特徴マップに再編成する。下図の例では、4 x 4の1channelの特徴マップを、色ごとに別の特徴マップとして切り出し、2 x 2の4channelの低解像度特徴マップに変換する。
在CNN中，随着层变深，特征图的分辨率降低，因此在YOLOv 2中，为了预测更准确的锚盒坐标，从高分辨率层采样的特征图在较低分辨率层和通道之间划分。 它结合在一起。 通常，除非要素图的大小匹配，否则无法组合通道，但在YOLOv 2中，您使用称为重组的独特技术将一个高分辨率要素图重新组织为多个低分辨率要素图。 在下面显示的示例中，4 x 4 1通道的特征映射被剪切为每种颜色的单独特征映射，并转换为4通道的2 x 2低分辨率特征映射。

<img src="data/reorg.png" width=320px>

<img src="data/reorg_1.png" width=160px>
<img src="data/reorg_4.png" width=160px>
<img src="data/reorg_2.png" width=160px>
<img src="data/reorg_3.png" width=160px>





## multi-scale training
YOLOv2はFCN構造のため入力画像のサイズは可変。モデル構造をそのままで、学習時に複数サイズの画像を交互に入力する事でロバストなモデルに訓練できる(入力画像の32 x 32ピクセルが特徴マップ上の1ピクセルに対応するので、入力画像の幅と高さは必ず32の倍数にする)。
由于YOLOv 2具有FCN结构，因此输入图像的大小是可变的。 在模型结构完整的情况下，您可以通过在学习时交替输入多个尺寸的图像来训练到健壮的模型（因为输入图像的32 x 32像素对应于要素图上的一个像素，输入图像的宽度 高度必须是32的倍数。

<img src="data/320x320.png" width=160px>
<img src="data/352x352.png" width=176px>
<img src="data/384x384.png" width=192px>
<img src="data/416x416.png" width=208px>

<img src="data/448x448.png" width=224px>
<img src="data/480x480.png" width=240px>
<img src="data/512x512.png" width=256px>
