# Análisis de Pensiones Colombianas

Proyecto enfocado en descargar, depurar y explorar la serie histórica publicada por Datos Abiertos Colombia sobre el valor de la unidad de los fondos de pensiones obligatorias y cesantías (`resource_id = uawh-cjvi`). El repositorio combina scripts y notebooks para dejar un dataset confiable y listo para análisis descriptivos o modelado.

## Objetivos
- Automatizar la descarga completa del recurso público `uawh-cjvi`.
- Estandarizar fechas, valores numéricos y etiquetas de fondos/entidades.
- Detectar y documentar duplicados, outliers y otras anomalías de calidad.
- Generar subconjuntos listos para análisis focalizados por entidad o tipo de fondo.

## Dataset
- **Fuente:** Portal de Datos Abiertos Colombia.
- **Cobertura:** Valores de cuota/unidad para fondos de pensiones obligatorias y cesantías en diferentes entidades.
- **Formato original:** API Socrata (`json`) consultada en lotes de 50.000 filas.

## Requisitos y configuración
- Python 3.9+ con conexión a internet para el primer consumo del API.
- Librerías empleadas: `pandas`, `requests`, `numpy`, `matplotlib`, `jupyter` (opcional para notebooks).

Instalación sugerida:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas requests numpy matplotlib jupyter
```

## Estructura del repositorio
- `pensiones.py`: pipeline principal (descarga, limpieza, generación de derivados y exportación final).
- `colfondos.py`: variante centrada en extracción y particionamiento básico.
- `pensiones.ipynb`: cuaderno para análisis exploratorio y visualizaciones.
- `data/raw/`: archivos originales y subsets por entidad/fondo.
- `data/processed/`: dataset limpio `pensiones_limpio_final.csv` y `resumen_limpieza.csv` con métricas de ejecución.
- `data/pensiones`: placeholder para resultados adicionales o entregables del análisis (personalizable).

## Flujo de procesamiento
1. **Extracción:** descarga paginada desde `https://www.datos.gov.co/resource/uawh-cjvi.json` hasta obtener todas las filas disponibles.
2. **Limpieza y normalización:** conversión de columnas `fecha` y `valor_unidad`, eliminación de símbolos, homogenización de mayúsculas/espacios, y depuración de duplicados exactos/conceptuales.
3. **Calidad de datos:** cálculo de porcentajes de valores nulos, cardinalidades y consistencia entre códigos y nombres de fondos/entidades.
4. **Detección de outliers:** método IQR sobre `valor_unidad`, creación de la bandera `es_outlier` para análisis posterior.
5. **Variables derivadas:** generación de `año`, `mes`, `trimestre` y clasificación `tipo_fondo` (Cesantías, Pensiones, Alternativo, Otros).
6. **Exportación:** guarda el dataset final en `data/processed/pensiones_limpio_final.csv`, el resumen operativo en `data/processed/resumen_limpieza.csv` y subconjuntos específicos en `data/raw/`.

## Cómo ejecutar

```bash
python pensiones.py
```

El script crea las carpetas necesarias bajo `data/`, imprime métricas de control y deja listos los archivos mencionados en el flujo.

### Trabajo en notebooks
- `pensiones.ipynb`: replica el pipeline y extiende con visualizaciones (boxplots, series temporales, etc.).
- `Analisis_Pensiones_Colfondos.ipynb`: notebook de referencia ubicado en `~/Downloads/`, ideal para contrastar resultados con el dataset limpio.

## Productos generados
- `data/processed/pensiones_limpio_final.csv`: dataset consolidado para análisis.
- `data/processed/resumen_limpieza.csv`: métricas de ejecución (filas finales, outliers detectados, memoria).
- `data/raw/*.csv`: subsets por entidad (`Skandia`, `Protección`, `Porvenir`, `Colfondos`) y por tipo de fondo (cesantías, pensiones por riesgo, retiro programado, alternativo).

## Próximos pasos sugeridos
- Incorporar visualizaciones en `matplotlib` o `seaborn` para seguimiento mensual de `valor_unidad`.
- Documentar hallazgos principales (tendencias, outliers relevantes) en el notebook o en un informe aparte.
- Empaquetar dependencias en un `requirements.txt` o `pyproject.toml` para facilitar la reproducción del proyecto.

## Fuente de datos
- Portal de Datos Abiertos Colombia: [Valor cuota unidad de fondos de pensiones obligatorias y cesantías](https://www.datos.gov.co/resource/uawh-cjvi.json)
