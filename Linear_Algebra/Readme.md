# LIST

* ##### [좌표 변환과 행렬](#transformation-and-matrix)

* ##### [Shape 위주 계산의 유용성](#importance-of-shape)

* ##### [Conjugate Transpose Matrix - 공역전치행렬 또는 켤레전치행렬](#conjugate-transpose-matrix)

* ##### [Transpose Matrix - 전치행렬](#transpose-matrix)

* ##### [Determinant - 행렬식](#determinant)





## Reference

- 프로그래머를 위한 선형대수







# Transformation and Matrix

- ##### 좌표 변환은 '정방행렬 A를 곱한다'라는 형태로 쓸 수 있다. 단, A에는 역행렬이 존재한다.

- ##### 역행렬을 지니는 정방행렬 A를 곱하는 것은 '좌표 변환'이라고 해석할 수 있다.



#### ex) 

다음과 같이 2쌍의 기저(e,e')를 사용해 벡터 v를 다음 두 가지로 표현이 가능한 경우

<img src="../picture/la_tam_0.jpg" height="40" width="300">

v와 v'의 관계를 나타내보자.

우선 이때 e와 e'의 대응관계가 다음과 같을 경우.

<img src="../picture/la_tam_1.jpg" height="60" width="130">

다음과 같이 쓸 수 있다.

<img src="../picture/la_tam_2.jpg" height="50" width="110">

이를 행렬 형태로 나타내면

<img src="../picture/la_tam_3.jpg" height="50" width="200">

혹은

<img src="../picture/la_tam_4.jpg" height="55" width="200">



로 쓸 수 있다.

또한, 다음이 성립한다.

<img src="../picture/la_tam_5.jpg" height="80" width="100">



# Transpose Matrix

#### 표기법

- <img src="../picture/la_tm_0.jpg" height="30" width="40"> 

  

#### 성질

- <img src="../picture/la_tm_1.jpg" height="30" width="150"> 
-  <img src="../picture/la_tm_2.jpg" height="25" width="150"> 
- <img src="../picture/la_tm_3.jpg" height="35" width="280"> 
- ##### <img src="../picture/la_tm_4.jpg" height="30" width="80"> 단, D는 대각행렬이다.



<img src="../picture/la_tm_5.jpg" height="60" width="300">일 때, 다음을 만족하는 경우 B는 A의 전치행렬(Transpose Matrix)이다.



즉, <img src="../picture/la_tm_6.jpg" height="26" width="180"> 





# Conjugate Transpose Matrix

#### 표기법

-  <img src="../picture/la_ctm_0.jpg" height="25" width="40"> 



#### 성질

- <img src="../picture/la_ctm_1.jpg" height="28" width="100">  
- <img src="../picture/la_ctm_2.jpg" height="30" width="100">  
- <img src="../picture/la_ctm_3.jpg" height="28" width="115">  
- <img src="../picture/la_ctm_4.jpg" height="27" width="150">  



#### ex)

<img src="../picture/la_ctm_5.jpg" height="80" width="415">





# Importance Of Shape

#### 행렬 계산을 할 때 크기에 주목을 해야 한다.



#### ex1)

##### Q.<img src="../picture/x.jpg" height="30" width="30">와<img src="../picture/v.jpg" height="27" width="27">는 각각 n차원의 열 벡터(혹은 종 벡터)일 때, <img src="../picture/la_ios_0.jpg" height="30" width="180">를 계산해라.

일반적으로 계산하게 되면 shape은 (nxn)(nxn +(nxn))(nx1)으로 nxn이 많아지게 된다.

하지만 shape에 집중하게 되면 아래와 같이 가능하다.

<img src="../picture/la_ios_1.jpg" width="280" height="180">





#### ex2)

##### Q. A는 n차 정방행렬, b와 c는 n차 열 벡터(혹은 종 벡터) 일 때, 다음이 성립함을 증명해라.

<img src="../picture/la_ios_2.jpg" height="70" width="350">

A.

<img src="../picture/la_ios_3.jpg" height="30" width="200">이 성립해야 하므로

<img src="../picture/la_ios_4.jpg" height="140" width="330">  

여기서 <img src="../picture/la_ios_5.jpg" height="28" width="70">는 Scalar이므로 <img src="../picture/la_ios_6.jpg" height="26" width="100">로 두면 식은 다음과 같이 된다.

<img src="../picture/la_ios_7.jpg" height="190" width="320"> 





# Determinant

#### 표기법

- <img src="../picture/la_de_0.jpg" height="25" width="55">  

- ##### <img src="../picture/la_de_1.jpg" height="30" width="35">, 실수의 <img src="../picture/la_de_2.jpg" height="23" width="24">와는 다르게 결과값이 음수 일 수 있다.



#### 성질

- <img src="../picture/la_de_3.jpg" height="25" width="80">  

- <img src="../picture/la_de_4.jpg" height="33" width="205"> 

- <img src="../picture/la_de_5.jpg" height="50" width="150"> 

- ##### <img src="../picture/la_de_6.jpg" height="25" width="80">인 행렬은 역행렬이 존재하지 않는다.  <img src="../picture/la_de_7.jpg" height="23" width="80">이 참이 되어야 하기 때문이다.

- <img src="../picture/la_de_8.jpg" height="30" width="300">  

- <img src="../picture/la_de_9.jpg" height="33" width="400">  

- ##### 행렬 <img src="../picture/la_de_10.jpg" height="25" width="120">일 때, <img src="../picture/la_de_11.jpg" height="25" width="200">

-  <img src="../picture/la_de_12.jpg" height="30" width="120"> 

- ##### 다음과 같은 성질을 다중선형성이라고 한다.

  ##### 1.  <img src="../picture/la_de_13.jpg" height="30" width="300"> 

  ##### ex)

  <img src="../picture/la_de_14.jpg" height="80" width="320">

  ##### 2. <img src="../picture/la_de_15.jpg" height="30" width="484">  

  ##### ex)

  <img src="../picture/la_de_16.jpg" height="75" width="530">

- ##### 다음과 같은 성질을 교대성이라고한다.

  <img src="../picture/la_de_17.jpg" height="34" width="370"> 

  ##### ex)

  <img src="../picture/la_de_18.jpg" height="80" width="300">

  ##### 상이 역전된다고 해석이 가능하다.

- ##### <img src="../picture/la_de_19.jpg" height="25" width="150">, 여기서 n은 행렬 A의 차수.

- ##### <img src="../picture/la_de_20.jpg" height="75" width="180">로 표현되는 상삼각행렬(Upper triangular matrix) 또는 하삼각행렬 (lower triangular matrix) 일 경우 <img src="../picture/la_de_21.jpg" height="30" width="200">



#### 특징

- ##### 실수 행렬의 행렬식은 실수이다. 복소 행렬의 행렬식은 일반적으로 복소수이다.

- ##### 정방행렬이 아닌 행렬에서 행렬식은 정의되지 않는다.