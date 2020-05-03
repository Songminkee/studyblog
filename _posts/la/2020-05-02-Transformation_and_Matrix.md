---
title: 좌표 변환과 행렬
author: Monch
category: linear_algebra
layout: post
---

[선형대수 LIST로 가기](https://songminkee.github.io//linear_algebra/2030/05/02/list.html)

 

  

<h2><b>Transformation and Matrix</b></h2>

<br>

- ##### <b>좌표 변환은 '정방행렬 A를 곱한다'라는 형태로 쓸 수 있다. 단, A에는 역행렬이 존재한다.</b>

- ##### <b>역행렬을 지니는 정방행렬 A를 곱하는 것은 '좌표 변환'이라고 해석할 수 있다.</b>



<h4><b>ex)</b></h4>

다음과 같이 2쌍의 기저(e,e')를 사용해 벡터 v를 다음 두 가지로 표현이 가능한 경우

<img src="{{'assets/picture/la_tam_0.jpg' | relative_url}}" height="40" width="300">

v와 v'의 관계를 나타내보자.

우선 이때 e와 e'의 대응관계가 다음과 같을 경우.

<img src="{{'assets/picture/la_tam_1.jpg' | relative_url}}" height="60" width="130">

다음과 같이 쓸 수 있다.

<img src="{{'assets/picture/la_tam_2.jpg' | relative_url}}" height="50" width="110">

이를 행렬 형태로 나타내면

<img src="{{'assets/picture/la_tam_3.jpg' | relative_url}}" height="50" width="200">

혹은

<img src="{{'assets/picture/la_tam_4.jpg' | relative_url}}" height="55" width="200">



로 쓸 수 있다.

또한, 다음이 성립한다.

<img src="{{'assets/picture/la_tam_5.jpg' | relative_url}}" height="80" width="100">

