## LIST

* ##### [좌표 변환과 행렬](#transformation-and-matrix)

* ##### [전치행렬](#transpose-matrix)

* ##### 공역전치행렬

* ##### TIP

  ###### 

## Reference

- 프로그래머를 위한 선형대수









# Transformation and Matrix

- ##### 좌표 변환은 '정방행렬 A를 곱한다'라는 형태로 쓸 수 있다. 단, A에는 역행렬이 존재한다.

- ##### 역행렬을 지니는 정방행렬 A를 곱하는 것은 '좌표 변환'이라고 해석할 수 있다.



#### ex) 

다음과 같이 2쌍의 기저(e,e')를 사용해 벡터 v를 다음 두 가지로 표현이 가능한 경우
$$
\vec{v}=x\vec{e_x}+y\vec{e_y}=x'\vec{e'_x}+y'\vec{e'_y}
$$
v와 v'의 관계를 나타내보자.

우선 이때 e와 e'의 대응관계가 다음과 같을 경우.
$$
\vec{e'_x}=3\vec{e_x}-2\vec{e_y}\\\vec{e'_y}=-\vec{e_x}+\vec{e_y}
$$
다음과 같이 쓸 수 있다.
$$
x=3x'-y'\\
y=-x'+y'
$$
이를 행렬 형태로 나타내면
$$
\begin{pmatrix}
x\\y
\end{pmatrix}
=\begin{pmatrix}
3&-1\\-2&1
\end{pmatrix}
\begin{pmatrix}
x'\\y'
\end{pmatrix}
$$
혹은
$$
\begin{pmatrix}
x'\\y'
\end{pmatrix}
=\begin{pmatrix}
1&1\\2&3
\end{pmatrix}
\begin{pmatrix}
x\\y
\end{pmatrix}
$$
로 쓸 수 있다.

또한, 다음이 성립한다.
$$
v'=Av\\v=A'v'\\A'=A^{-1}
$$




# Transpose Matrix

$ A=\begin{pmatrix}
a_{00}&... &a_{0j}\\
...&&...\\
a_{i0}&...&a_{ij}
\end{pmatrix},B=\begin{pmatrix}
b_{00}&...&b_{0j}\\
...&&...\\
b_{i0}&...&b_{ij}
\end{pmatrix} $ 





