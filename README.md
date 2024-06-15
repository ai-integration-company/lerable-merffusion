# lerrable–merfusion 

Ссылка на развёрнутую версию: https://labre-tau.vercel.app/

В случае, если фронтенд по ссылке выше не работает, протестируйте пайплайн из ноутбука tests_lm.ipyb
## Описание

Проект представляет из себя пайплайн состоящий из трёх моделей. VLM и две модели для inpainting: общего назначения и специальная модель для генерации фонов из [статьи](https://huggingface.co/yahoo-inc/photo-background-generation).

Пайплайн принимает фотографию объекта, для которого необходимо сгенерировать фон и возвращает фон, фото объекта с изменённым в соответствии с фото размером и координаты вставки объекта в фон.

## Детали реализации

Фото объекта подаётся в VLM, которая генерирует промпт для inpainting модели, а также определяет размеры и расположение объекта на будущей картинке. После этого объект приводишься к соответствующему размеру и располагается на будущем фоне в соответствии с правилом третей.

Затем с помощью специальной модели для генерации фонов создаётся фон вокруг правильно расположенного объекта. После этого объект вырезается, и уже с помощью обычно модели для inpainting заполняется место, которое находилось под обьектом.

Такой подход позволяет получить готовый фон, по которому, при желании, можно двигать объект. При этом все модели из пайплайна могут быть заменены на более продвинутые аналоги, что делает данный подход масштабируемым.
