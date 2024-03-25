import os # для работы с файловой сисиемой
import numpy as np #для создания массива
from PIL import Image #
import csv # для работы с импортом в csv
import pandas as pd # для работы с таблицей экселя
import time # для работы с таймерами

start_time = time.time() # таймер

# Считываем названия файлов и формируем относительные пути
dir = 'ions'
dir_learn='learn'
dir_validate='validate'
filenames = os.listdir(dir)
filenames_learn=os.listdir(dir_learn)
filenames_validate = os.listdir(dir_validate)
paths = []
for file in filenames:
    paths.append(dir + f'/{file}')
paths_learn = []
for file in filenames_learn:
    paths_learn.append(dir_learn + f'/{file}')
paths_validate = []
for file in filenames_validate:
    paths_validate.append(dir_validate + f'/{file}')


'''
сначала обрабатываем 100 картинок для обучения и 
100 картинок для валидации классическим методом,
чтобы иметь массив для валидации. Запись в .txt файл
'''


def Classic(name,currentpath,descr):
    ions = [17, 57, 92, 130]  # номер строки, с которой начинается поиск i-го иона
    count = 0
    file = open(name, 'w')
    for path in currentpath:
        b = [0, 0, 0, 0]  # Инициализируем пустой вектор ионов
        # Открываем картинку, преобразуем её пиксели в оттенки серого и считываем её как массив
        img = Image.open(path)
        gray_img = img.convert('L')
        gray_arr = np.asarray(gray_img)
        arr_trimmed = gray_arr[:, 27:-24]  # выделяем центральную часть картинки
        # Поиск иона
        for j in range(len(ions)):
            for i in range(5):
                rowpix_validate = arr_trimmed[ions[j] + 1]  # для дальнейшего обучения нейросети
                max_num = max(rowpix_validate)  # Находим пиксель с наибольшим значением оттенка серого
                if max_num > 50:  # Проверка превышения порогового значения
                    b[j] = 1
        print(f'Обработкa иона {descr} {count + 1}:\t\tитоговый вектор - \t{b[0]} {b[1]} {b[2]} {b[3]}\t\t\tИмя файла: {path}')
        file.write(
            f'Обработка иона {descr} {count + 1}:   итоговый вектор -     {b[0]} {b[1]} {b[2]} {b[3]}\n')
        count += 1
    file.close()
    print(f'\nКонец предобработки массива {descr}...\n'+'_ '*75+'\n\n')

'''
сначала обрабатываем 100 картинок для обучения и 
100 картинок для валидации классическим методом,
чтобы иметь массив для валидации. Запись в .txt файл
'''

Classic('data-learn.txt',paths_learn,'обучения')
Classic('data-validate.txt',paths_validate,'валидации')
Classic('data-ions.txt',paths,'')
#обучение


'''
теперь оборабатываем оставшиеся ионы с помощью простой нейросети.
для этого обработаем все файлы из массива обучения с уже известными результатами
'''

#подготовка массива пикселей - основные картинки
for path in paths:
    # Открываем картинку, преобразуем её пиксели в оттенки серого и считываем её как массив
    img = Image.open(path)
    gray_img = img.convert('L')
    gray_arr = np.asarray(gray_img)
    arr_trimmed = gray_arr[:, 26:-25] # выделяем центральную часть картинки

'''
    for j in range(len(ions)):
        for i in range(5):
            row = arr_trimmed[ions[j] + i]
            max_num = max(row)  # Находим пиксель с наибольшим значением оттенка серого
            if max_num > 50:  # Проверка превышения порогового значения
                b[j] = 1
'''




'''
f = open('data-csv.txt', 'r')
# Запись результатов в csv файл
with open('labeled_ions_korabeli_12.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow('Number;Filename;Qubit 1 state;Qubit 2 state;Qubit 3 state;Qubit 4 state'.split(';'))
    for s in f:
        writer.writerow(c.strip() for c in s.strip().split(';'))
f.close()
'''

print(f'\nВремя выполнения программы: {round(time.time() - start_time, 3)} c.')
input('. . . press Enter to exit . . .')
