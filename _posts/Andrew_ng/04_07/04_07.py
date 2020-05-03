import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_boston
#   **CRIM**: 자치 시(town) 별 1인당 범죄율
#
#   **ZN**: 25,000 평방피트를 초과하는 거주지역의 비율
#
#   **INDUS**: 비소매상업지역이 점유하고 있는 토지의 비율
#
#   **CHAS**: 찰스강의 경계에 위치해 있으면 1, 그렇지 않으면 0
#
#   **NOX**: 10ppm당 농축 일산화질소
#
#   **RM**: 주택 1가구당 평균 방의 개수
#
#   **AGE**: 1940년 이전에 건축된 소유주택의 비율
#
#   **DIS**: 5개의 보스턴 직업센터까지의 접근성 지수
#
#   **RAD**: 방사형 도로까지의 접근성 지수
#
#   **TAX**: 10,000 달러 당 재산세율
#
#   **PTRATIO**: 자치 시(town)별 학생/교사 비율
#
#   **B**: 1000(Bk-0.63)^2, 여기서 Bk는 자치시별 흑인의 비율을 말함.
#
#   **LSTAT**: 모집단의 하위계층 비율(%)
#
#   **MEDV**: 본인 소유의 주택가격(중앙값) (단위: $1,000)
import pandas as pd

boston = load_boston()
x_label = boston["data"]
y_label = boston["target"]
data = pd.DataFrame(x_label,columns = boston["feature_names"])
data["MEDV"]= y_label
print(boston["feature_names"])
x = x_label[:,12]
points = np.arange(-20,20,1)

def cost_func(w,b):
    h = x*w+b
    cost = np.power(h-y_label,2)
    return np.mean(cost)/2

w,b = np.meshgrid(points,points)
y = np.zeros((40,40))
for i in range(40):
    for j in range(40):
        y[i,j] = cost_func(w[i,j],b[i,j])


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')

ax.scatter3D(w,b,y)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('y')
plt.title("test")
plt.show()