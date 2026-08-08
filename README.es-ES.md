

# Introducción

Este proyecto fue desarrollado específicamente para el conjunto de datos CN-Celeb, utilizando el marco de trabajo `ECAPA-TDNN + AAM-Softmax`.

## Rendimiento

|   Datos de entrenamiento   |    Datos de prueba     | Aumento | EER (%) | minDCF (0.01) | Umbral |
|:--------:|:-----------:|:-------:|:-------:|:-------------:|:---:|
| CN-2-dev | CN-1-trials |   No    |  11.7   |    0.4999     | Por probar |
| CN-2-dev | CN-1-trials |   Sí    |  9.9   |    0.4276     | 0.180 |
| CN-2-dev | StarRail |   No    |  18.13   |    0.7229     | 0.547 |
| StarRail | StarRail |   No    |  0.578   |    0.0388     | 0.168 |

# Inicio Rápido

## Preparación

### Datos
* CN-Celeb 1 [[Descárgalo aquí]](http://openslr.org/82/)
* CN-Celeb 2 [[Descárgalo aquí]](http://openslr.org/82/)
> Los datos originales de CN-Celeb están en formato FLAC. Dado que convertirlos ocuparía espacio adicional en el disco, se leerán directamente en formato FLAC para el entrenamiento.

### Preparación de metadatos
* Genera `train.csv` utilizando `build_datalist.py`.
* El archivo `trial.lst` proporcionado en el conjunto de datos tiene la extensión `.wav`, lo cual parece ser un error. Utiliza la función `create_cnceleb_trails` en `dataset.py` para generar un nuevo archivo `trial.lst`.

### Aumento de datos
* Descarga el conjunto de datos de ruidos RIRS [[Descárgalo aquí]](http://openslr.org/28/)
* Descarga el conjunto de datos MUSAN [[Descárgalo aquí]](http://openslr.org/17/)
* Extrae los conjuntos de datos en la ruta `augmented_data\`
* Utiliza `dataprep.py` para preprocesar los ruidos RIRS.
* Agrega la opción `--augmentation` al ejecutar, como se muestra a continuación:
```
python trainECAPAModel.py --augmentation
```

### Entorno

```
conda create -n cnceleb python=3.12.3
conda activate cnceleb
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```
> Si no puedes instalar con pip, prueba con los siguientes mirrors:
> * Tsinghua: https://pypi.tuna.tsinghua.edu.cn/simple/
> * Alibaba Cloud: http://mirrors.aliyun.com/pypi/simple/
> * Universidad de Ciencia y Tecnología de China: https://pypi.mirrors.ustc.edu.cn/simple/
> * Universidad de Ciencia y Tecnología de Huazhong: http://pypi.hustunique.com/
> * Universidad de Tecnología de Shandong: http://pypi.sdutlinux.org/
> * Douban: http://pypi.douban.com/simple/

## Entrenamiento
1. Configura las rutas correspondientes en `trainECAPAModel.py`.
2. Activa el entorno de conda: `conda activate cnceleb`
3. Ejecuta `python trainECAPAModel.py`
4. Proporciona varias opciones como `backend`, `link_method`, `backbone`, etc. Consulta la ayuda para más detalles.

## Pruebas
1. Configura la ruta de `initial_model` en el programa principal.
2. Ejecuta `python trainECAPAModel.py --eval`

## Análisis de resultados
Los resultados de las pruebas con aumento de datos se guardan en `score_label.pkl`. Puedes cargarlos con el siguiente código:
```
with open('score_label.pkl', 'rb') as f:
    ids_dict, ids_true, ids_false, revues_dict, revues_true, revues_false = pickle.load(f)
```

## Demo
1. Descarga los pesos preentrenados desde [release](https://github.com/ZhaoQinlao/ECAPA-TDNN-CNCeleb/releases).
2. Inicia el script con el comando `gradio demo_with_gradio.py` y abre el enlace correspondiente en tu navegador.

## Agradecimientos

Este proyecto está basado en las modificaciones de [PunkMale/ECAPA-TDNN-CNCeleb](https://github.com/PunkMale/ECAPA-TDNN-CNCeleb) y [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN), y toma como referencia [Lantian Li/Sunine](https://gitlab.com/csltstu/sunine).
