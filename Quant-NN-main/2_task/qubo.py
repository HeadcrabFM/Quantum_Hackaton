#Попытка в вывод QUBO матрицы. Неудачная..(
from pyqubo import Spin
import numpy as np
import pickle as pkl
#Инициализация спинов
x1,x2,x3,x4,x5,x6,x7,x8 = Spin('x1'), Spin('x2'), Spin('x3'), Spin('x4'), Spin('x5'), Spin('x6'), Spin('x7'), Spin('x8')
#Инициализация гамильтониана
H = (128*x1+64*x2+32*x3+16*x4+8*x5+4*x6+2*x7+x8-50)**2
model = H.compile()
#Получение коэффициентов qubo-матрицы
qubo, offset = model.to_qubo()
result=qubo.items()
data=list(result)
#Инициализация numpy массива
massiv=np.array([[data[27][1],data[32][1],data[5][1],data[31][1],data[28][1],data[29][1],data[6][1],data[11][1]],
                [data[32][1],data[13][1],data[4][1],data[3][1],data[14][1],data[22][1],data[8][1],data[33][1]],
                [data[5][1],data[4][1],data[25][1],data[18][1],data[12][1],data[24][1],data[23][1],data[26][1]],
                [data[31][1],data[3][1],data[18][1],data[10][1],data[9][1],data[30][1],data[17][1],data[1][1]],
                [data[28][1],data[14][1],data[12][1],data[9][1],data[21][1],data[2][1],data[34][1],data[20][1]],
                [data[29][1],data[22][1],data[24][1],data[30][1],data[2][1],data[35][1],data[7][1],data[16][1]],
                [data[6][1],data[8][1],data[23][1],data[17][1],data[34][1],data[7][1],data[15][1],data[19][1]],
                [data[11][1],data[33][1],data[26][1],data[1][1],data[20][1],data[16][1],data[19][1],data[0][1]]])
#Запись numpy массива в pickle
with open('qubo.pkl','wb') as f:
    pkl.dump(massiv,f)
