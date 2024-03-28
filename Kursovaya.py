import os  # для работы с файловой сисиемой
import numpy as np  # для создания массива
from PIL import Image  # библиотека для работы с изображением
import csv  # для работы с импортом в csv
import pandas as pd  # для работы с таблицей экселя
import time  # для работы с таймерами


# Функция считывания имён файлов и формирования относительных путей
def GetListOfFiles(targetfolder):
    filelist = os.listdir(targetfolder)
    filepaths = []
    for file in filelist:
        filepaths.append(targetfolder + f'/{file}')
    return filepaths

# Функция создания папок для хранения файлов вывода и самих файлов вывода, вызывается внутри функций обработки
def Output(outputfolder, outputfile):
    # создание папки для хранения результатов
    if os.path.exists(outputfolder):
        pass
    else:
        os.makedirs(outputfolder)
    outputfile = open(outputfolder + f'/{outputfile}', 'w')
    return outputfile

# Функция для обработки изображения классическим методом
def Classic(outputfolder, outputfile, inputfolder, descr):
    count = 0
    file=Output(outputfolder, outputfile)
    for path in inputfolder:
        b = [0, 0, 0, 0]  # Инициализируем пустой вектор ионов
        # Открываем картинку, преобразуем её пиксели в оттенки серого и считываем её как массив
        img = Image.open(path)
        gray_img = img.convert('L')
        gray_arr = np.asarray(gray_img)
        arr_trimmed = gray_arr[:, 27:-24]  # выделяем центральную часть картинки
        arr_ions_trimmed = np.asarray([])
        # Поиск иона
        for j in range(len(ions)):
            for i in range(5):
                rowpix_validate = arr_trimmed[ions[j] + 1]  # для дальнейшего обучения нейросети
                max_num = max(rowpix_validate)  # Находим пиксель с наибольшим значением оттенка серого
                if max_num > 50:  # Проверка превышения порогового значения
                    b[j] = 1
        # вывод на экран
        print(
            f'Обработкa иона {descr} {count + 1}:\t\t'
            f'итоговый вектор -\t{b[0]} {b[1]} {b[2]} {b[3]}\t\t\t'
            f'Имя файла: {path}')
        # запись в .txt файл
        file.write(
            f'Обработка иона {descr} {count + 1}:\t\t'
            f'итоговый вектор -\t{b[0]} {b[1]} {b[2]} {b[3]}\t\t\t'
            f'Имя файла: {path}\n')
        count += 1
    file.close()
    print(f'\nКонец предобработки массива {descr}...\n' + '_ ' * 75 + '\n\n')





def GetIonMatrix(outputfolder, outputfile_human, outputfile_AI, inputfolder, descr):
    '''
    Функция получения пиксельной матрицы квадратика изображения
    получаем участок 5х5 с цветовыми кодами каждого пикселя
    сохраняем как 4 списка по 25 элементов (4 цветовых матрицы
    областей иона 5х5) для каждой картинки
    '''
    # Создание папки для хранения входных данных в читаемом нейросетью виде
    count = 0
    file_human=Output(outputfolder, outputfile_human)
    file_AI=Output(outputfolder, outputfile_AI)
    for path in inputfolder:
        # Открываем картинку, преобразуем её пиксели в оттенки серого и считываем её как массив
        img = Image.open(path)
        gray_img = img.convert('L')
        gray_arr = np.asarray(gray_img)
        file_human.write(f'Имя файла: \t\t{path}\n')
        print(f'Имя файла: \t\t{path}\n')
        for i in range(4):
            output = []
            arr_trimmed = gray_arr[ions[i]:-(164 - ions[i] - 5), 29:-22]  # выделяем центральную часть картинки
            for j in range(5):
                for k in range(5):
                    output.append(arr_trimmed[j, k])
            print(f'Ион №{count+1}.{i + 1}:\t\t'
                  f'{output}')
            file_human.write(f'Картинка №{count + 1}, Ион №{i + 1}:\t\t'
                       f'{output}\n')
        print('- ' * 11)
        file_human.write(f'{'- ' * 11}\n')
        count += 1
    file_human.close()
    file_AI.close()
    print(f'\nКонец получения матриц пикселей иона для файлов в папке {descr}...\n{'_ ' * 75}\n\n')

# Функция обработки данных в читаемом нейросетью виде.. было бы круто конечно если бы это внутри гет ион матриксе происходило.

'''
сначала обрабатываем 100 картинок для обучения и 
100 картинок для валидации классическим методом,
чтобы иметь массив для валидации. Запись в .txt файл
'''

start_time = time.time()  # таймер
ions = [17, 57, 92, 130]  # номер строки, с которой начинается поиск i-го иона
barrier_value = 50  # пороговое значение цвета

# Получаем список имён файлов и путей к ним для папок обучения, валидации,
# а также папки с основным массивом картинок для обработки
paths, paths_learn, paths_validate = (GetListOfFiles('ions'),
                                      GetListOfFiles('learn'),
                                      GetListOfFiles('validate'))

# Создание папки для хранения данных обработки


Classic('data', 'classic-learn.txt', paths_learn, 'обучения')
Classic('data', 'classic-validate.txt', paths_validate, 'валидации')
Classic('data', 'classic-ions.txt', paths, 'для основной обработки')

GetIonMatrix('data', 'matrix-learn-human.txt',
             'matrix-learn-AI.txt', paths_learn, 'обучения')
GetIonMatrix('data', 'matrix-validate-human.txt',
             'matrix-validate-AI.txt', paths_validate, 'валидации')
GetIonMatrix('data', 'matrix-ions-human.txt',
             'matrix-ions-AI.txt', paths, 'для основной обработки')

'''
теперь оборабатываем основной массив ионов с помощью простой нейросети.
для этого обработаем все файлы из массива обучения с уже известными результатами/
У нас получается 100*4 = 400 строчек для обучения в массиве ообучения,
а также 100*4 = 400 строчек для валидации.
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
