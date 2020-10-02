---
title: Faster R-CNN
author: Monch
category: Object Detection_
layout: post
---

 2015년 ICCV에 발표된 "Fast R-CNN"은 이전 네트워크와 비교했을 때 비약적인 속도의 향상을 불러왔다. 하지만 아직 실시간으로 동작하기 어려운 속도를 보여준다. 이때 가장 많이 시간이 들었던 부분은 Selective search를 이용한 Region proposal 연산이었다. 본 논문에서는 anchor box 개념을 도입해 convolutional network로 Region proposal을 수행하는 Region Proposal Network(RPN)을 소개한다.

<br>

<h2>Faster R-CNN Arichtecture</h2>

<img src="{{'assets/picture/faster_rcnn.jpg' | relative_url}}">

Faster R-CNN은 다음과 같이 두개의 module로 나뉜다.

- Region Proposal Network (RPN)
- Fast R-CNN detector

두 개의 모듈은 하나의 Conv Net을 공유하고 있으며 RPN은 이전의 Selective search와 같은 역할을 해준다.

<br>

<h3>Region Proposal Network</h3>

<img src="{{'assets/picture/rpn.jpg' | relative_url}}">

RPN은 마지막 conv layer의 output에서 $$n \times n$$ 크기의 conv layer 연산을 수행한다(크기를 보존하기 위해 padding을 한다). 이후 $$ 1 \tiems 1 \tiems (2 \times k) $$, $$ 1 \tiems 1 \tiems (4 \times k)$$ 연산을 병렬로 수행한다. 여기서 $$k$$는 anchor box의 수이다. 본 논문에서는 n을 3으로 설정했고 3개의 비율과 3개의 크기로 총 9개의 anchor box를 정의했다(k=9).



여기서 positive anchor box의 기준은 두가지 이다.

- ground-truth box와의 IOU가 가장 높은 anchor
- IOU가 0.7 이상인 anchor

본 논문에서는 두 번째 경우 드물게 positive sample을 찾지 못해 첫 번째도 정의 했다고 한다. 추가적으로 non-positive anchor중 0.3 이하의 IOU를 가지는 영역은 negative sample로 설정되었고 나머지 anchor는 objective function에 참여하지 않는다. 이 두 영역은 나중에 다시 ROI pooling을 통해 Fast R-CNN Classifier의 입력으로 들어간다. 동일한 ground truth에는 여러개의 anchor box가 설정될 수 있고 NMS 과정을 거쳐서 positive와 negative의 비율이 1:1로 되게 설정한다고 한다.



RPN의 loss function은 다음과 같다.


$$
L(\left\{p_{i}\right\},\left\{t_{i}\right\}) = \frac{1}{N_{cls}} \sum_{i} L_{cls}(p_{i},p_{i}^{*}) + \lambda \frac{1}{N_{reg}} \sum_{i} p_{i}^{*}L_{reg}(t_{i},t_{i}^{*})
$$


여기서 $$N_{cls}$$는 bach_size이며, $$N_{reg}$$는 모든 anchor의 수이다. 논문에서는 각각 256, (256 x 9)로 설정 되었다. $$\lambda$$는 두 term의 균형을 맞추기 위한 하이퍼 파라미터인데 본 논문에서는 10으로 설정되었다. $$L_{cls}$$는 오브젝트인지 아닌지에 대한 loss이고 $$L_reg$$는 bounding box에 대한 loss이다(Fast R-CNN과 완전 동일). $$L_{cls}$$ 식은 아래와 같다.


$$
L_{cls}(p_i,p^{*}) = -p^{*}_{i}log(p_{i}) - (1-p^{*}_{i})log(1-p_{i})
$$


논문 초반부에 계속 attention을 도입했다고 하는데 binary crossentropy를 통해 attention을 적용했다는 것 같다.





<br>

<h3>Reference</h3>

[1] S. Ren, K. He, R. Girshick. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In NIPS, 2015.

