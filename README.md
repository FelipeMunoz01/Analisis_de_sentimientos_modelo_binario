# Analisis_de_sentimientos_modelo_binario
Análisis de sentimientos mediante modelo binario


# Análisis de Sentimientos — Modelo Binario

Repositorio con implementación de un modelo binario para análisis de sentimientos (positivo vs. negativo). Este proyecto incluye preprocesamiento de texto, entrenamiento de un modelo sencillo, evaluación y scripts para inferencia.

## Contenido
- `data/` — Conjuntos de datos (raw y procesados).
- `notebooks/` — Notebooks exploratorios y de experimentación.
- `src/` — Código fuente (preprocesamiento, entrenamiento, evaluación, inferencia).
- `models/` — Modelos entrenados y pesos guardados.
- `requirements.txt` — Dependencias del proyecto.
- `README.md` — Este archivo.
- `LICENSE` — Licencia del proyecto (MIT).

## Requisitos
- Python 3.8+
- Recomendado: entorno virtual (venv o conda)

Instalar dependencias:
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Estructura de los datos
Se espera la siguiente organización de `data/`:
```
data/
  raw/
    train.csv
    test.csv
  processed/
    train_processed.csv
    test_processed.csv
```
Formato esperado de los CSV: columnas `text` (string) y `label` (0 o 1).

## Preprocesamiento
En `src/preprocess.py` se incluyen las funciones para:
- Normalizar texto (minúsculas, remover caracteres especiales).
- Tokenización básica o con librería (ej. NLTK, SpaCy).
- Remoción de stopwords (opcional).
- Vectorización (TF-IDF o embeddings).

Ejecutar preprocesamiento (ejemplo):
```bash
python src/preprocess.py --input data/raw/train.csv --output data/processed/train_processed.csv
```

## Entrenamiento
Se provee un script de ejemplo en `src/train.py`. Soporta:
- Modelos basados en TF-IDF + clasificador (Logistic Regression, SVM).
- Modelos con embeddings (opcional).

Ejecutar entrenamiento:
```bash
python src/train.py --train data/processed/train_processed.csv --model-dir models/ --epochs 5 --batch-size 32
```

Al finalizar, los artefactos (pesos, tokenizador/vectorizer, métricas) se guardan en `models/`.

## Evaluación
El script `src/evaluate.py` carga un modelo entrenado y evalúa sobre un conjunto de prueba, generando métricas:
- Accuracy
- Precision, Recall, F1
- Matriz de confusión

Ejemplo:
```bash
python src/evaluate.py --model models/mi_modelo.pkl --test data/processed/test_processed.csv
```

## Inferencia / Predicción
Script `src/predict.py` para realizar predicciones sobre textos nuevos:
```bash
python src/predict.py --model models/mi_modelo.pkl --text "Me encanta este producto"
# o con archivo
python src/predict.py --model models/mi_modelo.pkl --input-file examples/input_texts.txt --output-file examples/predictions.csv
```

## Notebooks
En `notebooks/` hay ejemplos de:
- Análisis exploratorio de datos (EDA).
- Pipeline de entrenamiento paso a paso.
- Visualización de resultados.

## Buenas prácticas reproducibles
- Fijar semillas aleatorias en entrenamiento para reproducibilidad.
- Guardar hiperparámetros y logs (ej. en `models/mi_modelo/params.json` y `models/mi_modelo/logs.txt`).
- Usar `requirements.txt` o `environment.yml` para controlar dependencias.

## Resultados esperados
Incluye una breve tabla de métricas de referencia (reemplazar con resultados reales tras experimentar):
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-score: 0.85

## Mejoras posibles
- Usar embeddings preentrenados (FastText, GloVe, BERT).
- Balanceo de clases (oversampling, weighting).
- Optimización de hiperparámetros (GridSearch / Optuna).
- Pipeline de despliegue (FastAPI, Docker).

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. 
## Contacto
Autor: FelipeMunoz01
