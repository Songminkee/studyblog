---
title: 히스토그램 (Histogram)
author: Monch
category: Computer Vision
layout: post
---

[목록으로가기](https://songminkee.github.io/cv/2030/05/03/list.html)

<h3>히스토그램</h3>

히스토그램이란 영상의 명암값이 나타난 빈도수로, [0,L-1] 사이의 명암값 각각이 영상에 몇 번 나타나는지를 표시한다.
Opencv를 이용해 이미지를 불러오는 경우 0~255로 명암값을 나타낸다. 즉 [0,255] 사이의 명암값 각각이 영상에 몇 번 나타나는지를 표시한다.
히스토그램의 여러 용도 중 하나는 영상의 특성을 파악하는 것이다. 히스토그램이 왼쪽으로 치우처져 있으면 어두운 영상, 오른쪽으로 치우처져 있으면 밝은 영상이다.

<h3>코드 구현</h3>

[코드링크](https://github.com/Songminkee/computer_vision/blob/master/histogram.ipynb)

코드의 구현은 간단하다. 각 픽셀의 노출 횟수를 카운트하면 된다.
예제는 유명한 레나사진을 사용할 것이다. OpenCV의 함수를 이용하면 bgr 순서로 채널이 되어 있기때문에 cv2의 imshow를 이용하는 게 아닐때는 rgb로 순서를 바꾸는게 좋다.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
img = cv2.imread('./data/lena.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()
```

<img src="{{'assets/picture/histogram_ex1.jpg' | relative_url}}">

numpy를 이용해 0으로 초기화된 리스트를 선언하고 각 채널의 명암값을 카운트한다.

```python
r_hist = np.zeros((256))
b_hist = np.zeros((256))
g_hist = np.zeros((256))

im_flat = np.reshape(img.copy(),(-1,3))
leng = len(im_flat)
for i in range(leng):
    b_hist[im_flat[i][0]]+=1
    g_hist[im_flat[i][1]]+=1
    r_hist[im_flat[i][2]]+=1
```

matplotlib을 이용하면 count된 횟수를 볼 수 있다.

```python
fig = plt.figure()
plt.subplot(131)
plt.plot(b_hist,color='b')
plt.xlim([0,256])

plt.subplot(132)
plt.plot(g_hist,color='g')
plt.xlim([0,256])

plt.subplot(133)
plt.plot(r_hist,color='r')
plt.xlim([0,256])
fig.tight_layout() # 그래프가 겹치는 것을 방지 할 수 있음
plt.show()
```



OpenCV에서는 cv2.calcHist로 단숨에 계산할 수 있다.

<img src="{{'assets/picture/histogram_equal_ex2.jpg' | relative_url}}">

누적 히스토그램은 다음과 같이 순서대로 이전의 값을 더하면 된다. 그리고 이 값을 정규화해 $$L-1$$을 곱해준 후 반올림 한다.

```python
## using cv
fig = plt.figure()
color = ['b','g','r']
for i in range(3):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.subplot(131+i)
    plt.plot(hist,color=color[i])
    plt.xlim([0,256])
fig.tight_layout()
plt.show()
```

<img src="{{'assets/picture/histogram_equal_ex3.jpg' | relative_url}}">