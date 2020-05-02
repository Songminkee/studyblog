## LIST

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

![equation](https://latex.codecogs.com/gif.latex?%5C%5C%5Cvec%7Bv%7D%3Dx%5Cvec%7Be_x%7D&plus;y%5Cvec%7Be_y%7D%3Dx%27%5Cvec%7Be%27_x%7D&plus;y%27%5Cvec%7Be%27_y%7D)

v와 v'의 관계를 나타내보자.

우선 이때 e와 e'의 대응관계가 다음과 같을 경우.

![equation](https://latex.codecogs.com/gif.latex?%5C%5C%5Cvec%7Be%27_x%7D%3D3%5Cvec%7Be_x%7D-2%5Cvec%7Be_y%7D%5C%5C%5Cvec%7Be%27_y%7D%3D-%5Cvec%7Be_x%7D&plus;%5Cvec%7Be_y%7D)

다음과 같이 쓸 수 있다.

![euqation](https://latex.codecogs.com/gif.latex?%5C%5Cx%3D3x%27-y%27%5C%5C%20y%3D-x%27&plus;y%27)

이를 행렬 형태로 나타내면

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bpmatrix%7D%20x%5C%5Cy%20%5Cend%7Bpmatrix%7D%20%3D%5Cbegin%7Bpmatrix%7D%203%26-1%5C%5C-2%261%20%5Cend%7Bpmatrix%7D%20%5Cbegin%7Bpmatrix%7D%20x%27%5C%5Cy%27%20%5Cend%7Bpmatrix%7D)

혹은

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bpmatrix%7D%20x%27%5C%5Cy%27%20%5Cend%7Bpmatrix%7D%20%3D%5Cbegin%7Bpmatrix%7D%201%261%5C%5C2%263%20%5Cend%7Bpmatrix%7D%20%5Cbegin%7Bpmatrix%7D%20x%5C%5Cy%20%5Cend%7Bpmatrix%7D)

로 쓸 수 있다.

또한, 다음이 성립한다.

![equation](https://latex.codecogs.com/gif.latex?%5C%5Cv%27%3DAv%5C%5C%20v%3DA%27v%27%5C%5C%20A%27%3DA%5E%7B-1%7D)



# Transpose Matrix

#### 표기법

- ![](https://latex.codecogs.com/gif.latex?A%5ET) 

  

#### 성질

- ![equation](https://latex.codecogs.com/gif.latex?%28%28A%29%5ET%29%5ET%20%3D%20A) 
- ![equation](https://latex.codecogs.com/gif.latex?%28AB%29%5ET%3DB%5ET%20A%5ET) 
- ![equation](https://latex.codecogs.com/gif.latex?%28A%5E%7B-1%7D%29%5ET%20A%5ET%20%3D%20%28A%20A%5E%7B-1%7D%29%5ET%20%3D%20I%5ET%20%3D%20I) 
- ![equation](https://latex.codecogs.com/gif.latex?D%5ET%3DD) 단, D는 대각행렬이다.

![equation](https://latex.codecogs.com/gif.latex?A%3D%5Cbegin%7Bpmatrix%7D%20a_%7B00%7D%26...%20%26a_%7B0j%7D%5C%5C%20...%26%26...%5C%5C%20a_%7Bi0%7D%26...%26a_%7Bij%7D%20%5Cend%7Bpmatrix%7D%2CB%3D%5Cbegin%7Bpmatrix%7D%20b_%7B00%7D%26...%26b_%7B0j%7D%5C%5C%20...%26%26...%5C%5C%20b_%7Bi0%7D%26...%26b_%7Bij%7D%20%5Cend%7Bpmatrix%7D)일 때, 다음을 만족하는 경우 B는 A의 전치행렬(Transpose Matrix)이다.

![equation](https://latex.codecogs.com/gif.latex?if%28a_%7Bij%7D%3Db_%7Bji%7D%29%2C%20A%5ET%3DB)







# Conjugate Transpose Matrix

#### 표기법

-  ![](https://latex.codecogs.com/gif.latex?A%5E*) 



#### 성질

- ![equation](https://latex.codecogs.com/gif.latex?A%5E*%20%3D%5Cbar%7BA%5ET%7D) 
- ![equation](https://latex.codecogs.com/gif.latex?%28A%5E*%29%5E*%3DA) 
- ![equation](https://latex.codecogs.com/gif.latex?%28AB%29%5E*%3DB%5E*A%5E*) 
- ![equation](https://latex.codecogs.com/gif.latex?%28A%5E%7B-1%7D%29%5E*%3D%28A%5E*%29%5E%7B-1%7D) 



#### ex)

![equation](https://latex.codecogs.com/gif.latex?A%3D%5Cbegin%7Bpmatrix%7D2&plus;i%269-2i%264%5C%5C7%265&plus;5i%263%20%5Cend%7Bpmatrix%7D%20%5Crightarrow%20A%5E*%20%3D%20%5Cbegin%7Bpmatrix%7D2-i%267%5C%5C9&plus;2i%265-5i%5C%5C4%263%20%5Cend%7Bpmatrix%7D)







# Importance Of Shape

#### 행렬 계산을 할 때 크기에 주목을 해야 한다.



#### ex1)

##### Q.![equation](https://latex.codecogs.com/gif.latex?x)와 ![equation](https://latex.codecogs.com/gif.latex?v)는 각각 n차원의 열 벡터(혹은 종 벡터)일 때, ![equation](https://latex.codecogs.com/gif.latex?y%3Dxx%5ET%20%28I&plus;vv%5ET%29x)를 계산해라.

일반적으로 계산하게 되면 shape은 (nxn)(nxn +(nxn))(nx1)으로 nxn이 많아지게 된다.

하지만 shape에 집중하게 되면 아래와 같이 가능하다.

<img src="../picture/linear algebra_1.jpg" width="280" height="180">





#### ex2)

##### Q. A는 n차 정방행렬, b와 c는 n차 열 벡터(혹은 종 벡터) 일 때, 다음이 성립함을 증명해라.

![](https://latex.codecogs.com/gif.latex?%28A&plus;bc%5ET%29%5E%7B-1%7D%3DA%5E%7B-1%7D-%5Cfrac%7BA%5E%7B-1%7Dbc%5ET%20A%5E%7B-1%7D%7D%7B1&plus;c%5ET%20A%5E%7B-1%7Db%7D)

A.

![](https://latex.codecogs.com/gif.latex?%28A&plus;bc%5ET%29%5E%7B-1%7D%28A&plus;bc%5ET%29%3DI)이 성립해야 하므로

![](https://latex.codecogs.com/gif.latex?%5C%5C%28A&plus;bc%5ET%29%5E%7B-1%7D%3DA%5E%7B-1%7D-%5Cfrac%7BA%5E%7B-1%7Dbc%5ET%20A%5E%7B-1%7D%7D%7B1&plus;c%5ETA%5E%7B-1%7Db%7D%5C%5C%20%5CRightarrow%20%28A%5E%7B-1%7D-%5Cfrac%7BA%5E%7B-1%7Dbc%5ET%20A%5E%7B-1%7D%7D%7B1&plus;c%5ETA%5E%7B-1%7Db%7D%29%28A&plus;bc%5ET%29%5C%5C%20%5CRightarrow%20I&plus;A%5E%7B-1%7Dbc%5ET-%5Cfrac%7BA%5E%7B-1%7Dbc%5ET&plus;A%5E%7B-1%7Dbc%5ET%20A%5E%7B-1%7Dbc%5ET%7D%7B1&plus;c%5ET%20A%5E%7B-1%7Db%7D) 

여기서 ![](https://latex.codecogs.com/gif.latex?c%5ETA%5E%7B-1%7Db)는 Scalar이므로 ![](https://latex.codecogs.com/gif.latex?c%5ETA%5E%7B-1%7Db%3D%5Calpha)로 두면 식은 다음과 같이 된다.

![](https://latex.codecogs.com/gif.latex?%5C%5CI&plus;A%5E%7B-1%7Dbc%5ET-%5Cfrac%7BA%5E%7B-1%7Dbc%5ET&plus;A%5E%7B-1%7Dbc%5ETA%5E%7B-1%7Dbc%5ET%7D%7B1&plus;c%5ETA%5E%7B-1%7Db%7D%5C%5C%5C%5C%20%5CRightarrow%20I&plus;A%5E%7B-1%7Dbc%5ET-%5Cfrac%7BA%5E%7B-1%7Dbc%5ET&plus;%5Calpha%28A%5E%7B-1%7Dbc%5ET%29%7D%7B1&plus;%5Calpha%7D%5C%5C%5C%5C%20%5CRightarrow%20I&plus;A%5E%7B-1%7Dbc%5ET-%5Cfrac%7B%281&plus;%5Calpha%29%28A%5E%7B-1%7Dbc%5ET%29%7D%7B%281&plus;%5Calpha%29%7D%5C%5C%5C%5C%20%5CRightarrow%20I&plus;A%5E%7B-1%7Dbc%5ET-A%5E%7B-1%7Db%5ET%3DI) 





# Determinant

#### 표기법

- ![](https://latex.codecogs.com/gif.latex?det%20A) 

- ##### ![](https://latex.codecogs.com/gif.latex?%5Cleft%20%7C%20A%20%5Cright%20%7C) , 실수의 ![](https://latex.codecogs.com/gif.latex?%5Cleft%20%7C%20%5Ccdot%20%5Cright%20%7C)와는 다르게 결과값이 음수 일 수 있다.



#### 성질

- ![](https://latex.codecogs.com/gif.latex?detI%3D1) 

- ![](https://latex.codecogs.com/gif.latex?det%28AB%29%3D%28detA%29%28detB%29) 

- ![](https://latex.codecogs.com/gif.latex?det%28A%5E%7B-1%7D%29%3D%5Cfrac%7B1%7D%7BdetA%7D) 

- ##### ![](https://latex.codecogs.com/gif.latex?det%28A%5E%7B-1%7D%29%3D%5Cfrac%7B1%7D%7BdetA%7D)인 행렬은 역행렬이 존재하지 않는다.  ![](https://latex.codecogs.com/gif.latex?0%5Ctimes%20%3F%3D1)이 참이 되어야 하기 때문이다.

- ![](https://latex.codecogs.com/gif.latex?det%28diag%28a_1%2C%20...%20%2C%20a_n%29%29%3Da_1%5Ctimes...%5Ctimes%20a_n) 

- ![](https://latex.codecogs.com/gif.latex?det%28diag%28A_%7B1%7D%2C...%2CA_n%29%29%3Ddet%28A_1%29%5Ctimes...%5Ctimes%20det%28A_n%29) 

- ##### 행렬 ![](https://latex.codecogs.com/gif.latex?A%3D%28%5Cvec%7Ba_1%7D%2C%5Cvec%7Ba_2%7D%2C%5Cvec%7Ba_3%7D%29)일 때, ![](https://latex.codecogs.com/gif.latex?detA%3Ddet%28%5Cvec%7Ba_1%7D%2C%5Cvec%7Ba_2%7D%2C%5Cvec%7Ba_1%7D&plus;c%5Cvec%7Ba_2%7D%29)

- ![](https://latex.codecogs.com/gif.latex?detA%3DdetA%5ET) 

- ##### 다음과 같은 성질을 다중선형성이라고 한다.

  ##### 1. ![](https://latex.codecogs.com/gif.latex?det%28c%5Cvec%7Ba_1%7D%2C%5Cvec%7Ba_2%7D%2C...%2C%5Cvec%7Ba_n%7D%29%3Dcdet%28%5Cvec%7Ba_1%7D%2C%5Cvec%7Ba_2%7D%2C...%2C%5Cvec%7Ba_n%7D%29) 

  ##### ex)

  ![](https://latex.codecogs.com/gif.latex?%5C%5Cdet%5Cbegin%7Bpmatrix%7D1%2610%265%5C%5C1%2620%267%5C%5C1%2630%266%20%5Cend%7Bpmatrix%7D%3D10det%5Cbegin%7Bpmatrix%7D1%261%265%5C%5C1%262%266%5C%5C1%263%266%5Cend%7Bpmatrix%7D%5C%5C)

  ##### 2. ![](https://latex.codecogs.com/gif.latex?det%28%5Cvec%7Ba_1%7D&plus;%5Cvec%7Ba%27_1%7D%2C%5Cvec%7Ba_2%7D%2C...%2C%5Cvec%7Ba_n%7D%29%3Ddet%28%5Cvec%7Ba_1%7D%2C%5Cvec%7Ba_2%7D%2C...%2C%5Cvec%7Ba_n%7D%29&plus;det%28a%27_1%2C%5Cvec%7Ba_2%7D%2C...%2C%5Cvec%7Ba_n%7D%29) 

  ##### ex)

  ![](https://latex.codecogs.com/gif.latex?det%5Cbegin%7Bpmatrix%7D1%261%265%5C%5C1%262%266%5C%5C1%263%266%5Cend%7Bpmatrix%7D&plus;%5Cbegin%7Bpmatrix%7D1%261%265%5C%5C1%267%267%5C%5C1%261%266%5Cend%7Bpmatrix%7D%3D%5Cbegin%7Bpmatrix%7D1%26%281&plus;1%29%265%5C%5C1%26%282&plus;7%29%266%5C%5C1%26%283&plus;1%29%266%5Cend%7Bpmatrix%7D%3Ddet%5Cbegin%7Bpmatrix%7D1%262%265%5C%5C1%269%267%5C%5C1%264%266%5Cend%7Bpmatrix%7D)

- ##### 다음과 같은 성질을 교대성이라고한다.

  ![](https://latex.codecogs.com/gif.latex?det%28%5Cvec%7Ba_2%7D%2C%5Cvec%7Ba_1%7D%2C%5Cvec%7Ba_3%7D%2C...%2C%5Cvec%7Ba_n%7D%29%3D-det%28%5Cvec%7Ba_1%7D%2C%5Cvec%7Ba_2%7D%2C%5Cvec%7Ba_3%7D%2C...%2C%5Cvec%7Ba_n%7D%29) 

  ##### ex)

  ![](https://latex.codecogs.com/gif.latex?det%5Cbegin%7Bpmatrix%7D1%261%265%5C%5C1%262%267%5C%5C1%263%266%20%5Cend%7Bpmatrix%7D%3D-det%5Cbegin%7Bpmatrix%7D1%261%265%5C%5C2%261%267%5C%5C3%261%266%20%5Cend%7Bpmatrix%7D)

  ##### 상이 역전된다고 해석이 가능하다.

- ##### ![](https://latex.codecogs.com/gif.latex?det%28cA%29%3Dc%5EndetA%5C%5C) , 여기서 n은 행렬 A의 차수.

- ##### ![](https://latex.codecogs.com/gif.latex?A%3D%5Cbegin%7Bpmatrix%7Da_%7B11%7D%26a_%7B12%7D%26a_%7B13%7D%5C%5C0%26a_%7B22%7D%26a_%7B23%7D%5C%5C0%260%26a_%7B33%7D%20%5Cend%7Bpmatrix%7D)로 표현되는 상삼각행렬(Upper triangular matrix) 또는 하삼각행렬 (lower triangular matrix) 일 경우 ![](https://latex.codecogs.com/gif.latex?det%20A%3Da_%7B11%7D%5Ctimes%20a_%7B22%7D%5Ctimes%20a_%7B33%7D)



#### 특징

- ##### 실수 행렬의 행렬식은 실수이다. 복소 행렬의 행렬식은 일반적으로 복소수이다.

- ##### 정방행렬이 아닌 행렬에서 행렬식은 정의되지 않는다.