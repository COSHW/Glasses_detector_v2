Проект на автомат

Для работы необходим Anaconda python 3.7 с установленным tensorflow==1.4.0, opencv==3.4.1.


Запустить через Anaconda Powershell Prompt в папке проекта на примерах: 

Для одной или нескольних фотографий: python .\main.py --image=Пути к файлу(ам), разделенные ", "

Для датасета из фотографий, состоящего из двух папок людей в очках и людей без очков: python .\predict_dataset.py --dataset_fgolder=Путь к датасету


Готовые примеры: 
python .\main.py --image="images\FaceWithoutGlasses\18.01anne-hathaway.jpg, images\FaceWithoutGlasses\18.pexels-photo-1222271.jpeg, images\FaceWithGlasses\image2 (204).PNG"

python .\predict_dataset.py --dataset_folder=test_dataset


В папке images находятся изображения, которые использовались для тренеровки модели.

В папке saved_data находятся датасет с матрицами картинок, датасет с лейблами, натренерованная модель и файл для распознования лица.

Код face_detector.py использвуется для того, чтобы найти и вырезать в изображении лицо человека.

Код training_data.py используется для создания датасетов матриц изображений X.pickle и лейблов y.pickle.

Код training.py используется для тренировки модели model.h5, используя датасеты матриц изображений X.pickle и лейблов y.pickle.

Код get_model_accuracy.py используется для получения точности модели model.h5.

Код main.py используется для получения прогноза модели model.h5 о наличии на фотографии(ях) человека в очках.

Код predict_dataset.py используется для показа точности модели на выбранном датасете и вывод предположения модели о фотографиях выбранного датасета.
