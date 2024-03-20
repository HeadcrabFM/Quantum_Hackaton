import os
import numpy as np
from PIL import Image
import csv
import time

start_time = time.time()
#Считываем названия файлов и формируем относительные пути
dir='pics'
filenames=os.listdir(dir)
paths=[]
for file in filenames:
    paths.append(dir+f'/{file}')
    
ions=[17,57,92,130]#номер строки, с которой начинается поиск i-го иона
count=0
file = open('data-csv.txt', 'w')
file1 = open('data-clear.txt', 'w')

for path in paths:
    b=[0,0,0,0] #Инициализируем вектор ионов
    #Открываем картинку, преобразуем её пиксели в оттенки серого и считываем её как массив
    img = Image.open(path)
    gray_img = img.convert('L') 
    gray_arr = np.asarray(gray_img)
    #Поиск иона
    for j in range (len(ions)):
        for i in range(5):
            row = gray_arr[ions[j]+i]
            max_num=max(row)#Находим пиксель с наибольшим значением оттенка серого
            if max_num>50:#Проверка превышения порогового значения
                b[j]=1
        print(f'Обработка иона {count+1}:   итоговый вектор - {b[0]} {b[1]} {b[2]} {b[3]}')

    file.write(f'{count+1};{path[5:]};{b[0]};{b[1]};{b[2]};{b[3]}\n') # Подготовка данных для записи в читаемом виде
    file1.write(f'Обработка иона {count+1}:   итоговый вектор -     {b[0]} {b[1]} {b[2]} {b[3]}\n')  # Подготовка данных для записи в csv
    count+=1
file.close()

f = open('data-csv.txt', 'r')
#Запись результатов в csv файл
with open('labeled_ions_korabeli_12.csv', 'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow('Number;Filename;Qubit 1 state;Qubit 2 state;Qubit 3 state;Qubit 4 state'.split(';'))
    for s in f:
        writer.writerow(c.strip() for c in s.strip().split(';'))
f.close()
print(f'\nВремя выполнения программы: {round(time.time()-start_time,3)} c.')
input('. . . press Enter to exit . . .')
