---
title: 가우스 요르단 소거법
author: Monch
category: linear_algebra
layout: post
---

[선형대수 LIST로 가기](https://songminkee.github.io/linear_algebra/2030/05/03/list.html)

  

 <h2><b>가우스 소거법과 가우스 요르단 소거법</b></h2> 

<br>

다음 세 가지의 기본 행 연산을 통해 연립일차방정식의 첨가행렬(계수행렬과 상수행렬을 묶은 행렬)을 기약 행 사다리꼴로 변환하여 해를 구한다.

1) 한 행을 상수배 한다.

2) 한 행을 상수배하여 다른 행에 더한다.

3) 두 행을 맞바꾼다.

먼저 가우스 소거법부터 살펴보자.

<br>

<h3><b>연립 방정식</b></h3>

다음과 같은 연립방정식을 생각해 보자
$$
2x_1+3x_2+3x_3=9\\3x_1+4x_2+2x_3=9\\-2x_1-2x_2+3x_3=2
$$
이는 2개의 행렬로 나타내면 다음과 같이 나타낼 수 있다.
$$
A=\begin{pmatrix}2&3&3\\3&4&2\\-2&-2&3\end{pmatrix},  \quad y=\begin{pmatrix}9\\9\\2\end{pmatrix}
$$
가우스 소거법에서는 위의 행렬을 하나의 블록 행렬로 나타낸다.
$$
\left(
  \begin{array}{ccc|c}
  2 & 3 & 3 & 9 \\
  3&4&2&9 \\
  -2&-2&3&2
\end{array} \right)\left(\begin{array}{}x_1\\x_2\\ x_3 \\\hline 1 \end{array}\right)=\begin{pmatrix}0\\0\\0\end{pmatrix}
$$
여기서 각 행의원소의 앞부분을 소거한다. 과정을 나타내면 다음과 같다.
$$
\left(
  \begin{array}{ccc|c}
  (1) & \frac{3}{2} & \frac{3}{2} & \frac{9}{2} \\
  3 & 4 & 2 & 9 \\
  -2&-2&3&2
\end{array} \right) \left(\begin{array}{}x_1\\x_2\\ x_3 \\\hline 1 \end{array}\right)=\begin{pmatrix}0\\0\\0\end{pmatrix}\\
\Rightarrow
\left(
  \begin{array}{ccc|c}
  1 & \frac{3}{2} & \frac{3}{2} & \frac{9}{2} \\
  (0) & -\frac{1}{2} & -\frac{5}{2} & -\frac{9}{2} \\
  (0)&1&6&11
\end{array} \right) \left(\begin{array}{}x_1\\x_2\\ x_3 \\\hline 1 \end{array}\right)=\begin{pmatrix}0\\0\\0\end{pmatrix}\\
\Rightarrow 
\left(
  \begin{array}{ccc|c}
  1 & \frac{3}{2} & \frac{3}{2} & \frac{9}{2} \\
  0 & (1) & 5& 9 \\
  0&1&6&11
\end{array} \right) \left(\begin{array}{}x_1\\x_2\\ x_3 \\\hline 1 \end{array}\right)=\begin{pmatrix}0\\0\\0\end{pmatrix}\\
\Rightarrow 
\left(
  \begin{array}{ccc|c}
  1 & \frac{3}{2} & \frac{3}{2} & \frac{9}{2} \\
  0 & 1 & 5& 9 \\
  0&(0)&1&2
\end{array} \right)\left(\begin{array}{}x_1\\x_2\\ x_3 \\\hline 1 \end{array}\right)=\begin{pmatrix}0\\0\\0\end{pmatrix}\\
$$
여기까지의 과정을 가우스 소거법이라고 한다. 가우스 요르단(Gauss jordan) 소거법은 여기서 한단계 더 나아간다.
$$
\left(
  \begin{array}{ccc|c}
  1 & \frac{3}{2} & \frac{3}{2} & \frac{9}{2} \\
  0 & 1 & 5& 9 \\
  0&(0)&1&2
\end{array} \right) \left(\begin{array}{}x_1\\x_2\\ x_3 \\\hline 1 \end{array}\right)=\begin{pmatrix}0\\0\\0\end{pmatrix}\\
\Rightarrow 
\left(
  \begin{array}{ccc|c}
  1 & \frac{3}{2} & \frac{3}{2} & \frac{9}{2} \\
  0 & 1 & (0)& -1 \\
  0&0&1&2
\end{array} \right) \left(\begin{array}{}x_1\\x_2\\ x_3 \\\hline 1 \end{array}\right)=\begin{pmatrix}0\\0\\0\end{pmatrix}\\
\Rightarrow 
\left(
  \begin{array}{ccc|c}
  1 & (0) & \frac{3}{2} & 6 \\
  0 & 1 & 0& -1 \\
  0&0&1&2
\end{array} \right) \left(\begin{array}{}x_1\\x_2\\ x_3 \\\hline 1 \end{array}\right)=\begin{pmatrix}0\\0\\0\end{pmatrix}\\
\Rightarrow 
\left(
  \begin{array}{ccc|c}
  1 & 0 & (0) & 3 \\
  0 & 1 & 0& -1 \\
  0&0&1&2
\end{array} \right) \left(\begin{array}{}x_1\\x_2\\ x_3 \\\hline 1 \end{array}\right)=\begin{pmatrix}0\\0\\0\end{pmatrix}\\
$$
이렇게 각 행이 바로 각 $$x_{i}$$의 해를 나타낸다.



<h3><b>역행렬</b></h3>

행렬 $$A$$의 역행렬 $$A^{-1}$$에 대한 관계를 다음과 같이 나타내자
$$
A^{-1}(A|I)=(I|A^{-1})
$$
이 식을 해석하면 가우스 소거법으로 A를 I로 만들면 우측에는 A의 역행렬이 나타낸다로 이해할 수 있다.

자세한 계산은 다음과 같이 계산된다
$$
(A|I)\\
\Rightarrow 
\left(
  \begin{array}{ccc|ccc}
  2 & 3 & 3 & 1 & 0 & 0 \\
  \hline
  3 & 4 & 2 & 0 & 1 & 0\\
  \hline
  -2 & -2 & 3 & 0 & 0 & 1
\end{array} \right)\\
\Rightarrow 
\left(
  \begin{array}{ccc|ccc}
  (1) & \frac{3}{2} & \frac{3}{2} & \frac{1}{2} & 0 & 0 \\
  \hline
  3 & 4 & 2 & 0 & 1 & 0\\
  \hline
  -2 & -2 & 3 & 0 & 0 & 1
\end{array} \right)\\
\Rightarrow 
\left(
  \begin{array}{ccc|ccc}
  1 & \frac{3}{2} & \frac{3}{2} & \frac{1}{2} & 0 & 0 \\
  \hline
  (0) & -\frac{1}{2} & -\frac{5}{2} & -\frac{3}{2} & 1 & 0\\
  \hline
  (0) & 1 & 6 & 1 & 0 & 1
\end{array} \right)\\
\Rightarrow 
\left(
  \begin{array}{ccc|ccc}
  1 & \frac{3}{2} & \frac{3}{2} & \frac{1}{2} & 0 & 0 \\
  \hline
  0 & (1) & 5 & 3 & -2 & 0\\
  \hline
  0 & 1 & 6 & 1 & 0 & 1
\end{array} \right)\\
\Rightarrow 
\left(
  \begin{array}{ccc|ccc}
  1 & (0) & -6 & -4 & 3 & 0 \\
  \hline
  0 & 1 & 5 & 3 & -2 & 0\\
  \hline
  0 & (0) & 1 & -2 & 2 & 1
\end{array} \right)\\
\Rightarrow 
\left(
  \begin{array}{ccc|ccc}
  1 & 0 & (0) & -16 & 15 & 6 \\
  \hline
  0 & 1 & (0) & 13 & -12 & -5\\
  \hline
  0 & 0 & 1 & -2 & 2 & 1
\end{array} \right)\\
=(I|A^{-1})
$$
이렇게 바로 $$A$$의 역행렬을 구할 수 있다.

물론 라이브러리에 구현이 되어 있어 손으로 계산할 일이 없지만 손 계산이 필요할 때 기억해 두면 좋을 것 같다.

