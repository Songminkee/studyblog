---
title: Shape의 중요성
author: Monch
category: linear_algebra
layout: post
---

### [선형대수 LIST](https://songminkee.github.io//linear_algebra/2020/05/02/list.html)



 <h1><b>Importance of Shape</b></h1>

<h3><b>행렬 계산을 할 때 크기에 주목을 해야 한다!!</b></h3>

<h3><b>아래의 예시를 보며 맘에 새기자</b></h3>



<h3><b>ex1)</b></h3>

##### <b>Q.<img src="{{'assets/picture/x.jpg' | relative_url}}" height="30" width="30">와<img src="{{'assets/picture/v.jpg' | relative_url}}" height="27" width="27">는 각각 n차원의 열 벡터(혹은 종 벡터)일 때, <img src="{{'assets/picture/la_ios_0.jpg' | relative_url}}" height="30" width="180">를 계산해라.</b>

일반적으로 계산하게 되면 shape은 (nxn)(nxn +(nxn))(nx1)으로 nxn이 많아지게 된다.

하지만 shape에 집중하게 되면 아래와 같이 가능하다.

<img src="{{'assets/picture/la_ios_1.jpg' | relative_url}}" width="280" height="180">





<h3><b>ex2)</b></h3>

##### <b>Q. A는 n차 정방행렬, b와 c는 n차 열 벡터(혹은 종 벡터) 일 때, 다음이 성립함을 증명해라.</b>

<img src="{{'assets/picture/la_ios_2.jpg' | relative_url}}" height="70" width="350">

A.

<img src="{{'assets/picture/la_ios_3.jpg' | relative_url}}" height="30" width="200">이 성립해야 하므로

<img src="{{'assets/picture/la_ios_4.jpg' | relative_url}}" height="140" width="330">  

여기서 <img src="{{'assets/picture/la_ios_5.jpg' | relative_url}}" height="28" width="70">는 Scalar이므로 <img src="{{'assets/picture/la_ios_6.jpg' | relative_url}}" height="26" width="100">로 두면 식은 다음과 같이 된다.

<img src="{{'assets/picture/la_ios_7.jpg' | relative_url}}" height="190" width="320"> 


