import pandas as pd
import requests
import time
import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path
import os

BASE = "https://www.datos.gov.co"
RESOURCE = "uawh-cjvi"  # ID del dataset
URL = f"{BASE}/resource/{RESOURCE}.json"

try:
    total_filas = int(requests.get(f"{URL}?$select=count(*)").json()[0]["count"])
except Exception:
    total_filas  = None
print("Total reportado:", total_filas )

Lista_paginas = []
limit = 50000         
offset = 0
while True:
    params = {"$limit": limit, "$offset": offset}
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    respuestaJson = r.json()
    if not respuestaJson: # fin de datos
        break
    Lista_paginas.append(pd.DataFrame(respuestaJson))
    offset += limit
    print(f"Descargadas: {offset} filas…")
    time.sleep(0.3)     # pequeña pausa para no saturar

if Lista_paginas:  # Si la lista NO está vacía
    df = pd.concat(Lista_paginas, ignore_index=True)
else:              # Si la lista está vacía
    df = pd.DataFrame()

# fechas (ajusta el nombre si difiere)
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

# numérico (quita comas/puntos si vienen como texto con separadores)
df["valor_unidad"] = (
    df["valor_unidad"]
      .astype(str)
      .str.replace(r"[^\d\-,\.]", "", regex=True)  # limpia símbolos
      .str.replace(",", ".", regex=False)          # si usan coma decimal
      .astype(float)
)

print(df.dtypes)

# % de nulos por columna
nulls = df.isna().mean().sort_values(ascending=False).mul(100).round(2)
print(nulls)

cardinalidad = df.nunique(dropna=True).sort_values(ascending=False)
print(cardinalidad)

# valores únicos (muestra)
print("Valores únicos en nombre_entidad:", df["nombre_entidad"].dropna().unique()[:10])
print("Valores únicos en nombre_fondo:", df["nombre_fondo"].dropna().unique()[:10])
print("Conteo nombre_entidad:")
print(df["nombre_entidad"].value_counts(dropna=False).head(10))
print("Conteo nombre_fondo:")
print(df["nombre_fondo"].value_counts(dropna=False).head(20))

df_limpio = df.drop(columns=["codigo_entidad", "codigo_patrimonio"])
df_limpio.to_csv("data/raw/pensionesLimpio.csv", index=False)

dict_entidad = (
    df[["nombre_entidad", "codigo_entidad"]]
    .drop_duplicates()
    .set_index("nombre_entidad")["codigo_entidad"]
    .to_dict()
)

dict_fondo = (
    df[["nombre_fondo", "codigo_patrimonio"]]
    .drop_duplicates()
    .set_index("nombre_fondo")["codigo_patrimonio"]
    .to_dict()
)

df[["nombre_entidad", "codigo_entidad"]].drop_duplicates() \
   .to_csv("data/raw/entidad_codigo.csv", index=False)

df[["nombre_fondo", "codigo_patrimonio"]].drop_duplicates() \
   .to_csv("data/raw/fondos_codigo.csv", index=False)

# cuántos names por código (debe ser 1 si es one-to-one)
print("Relación código_entidad → nombre_entidad:")
print(df.groupby("codigo_entidad")["nombre_entidad"].nunique().sort_values(ascending=False).head())
print("Relación código_patrimonio → nombre_fondo:")
print(df.groupby("codigo_patrimonio")["nombre_fondo"].nunique().sort_values(ascending=False).head())

# y al revés: cuántos códigos por nombre
print("Relación nombre_entidad → código_entidad:")
print(df.groupby("nombre_entidad")["codigo_entidad"].nunique().sort_values(ascending=False).head())
print("Relación nombre_fondo → código_patrimonio:")
print(df.groupby("nombre_fondo")["codigo_patrimonio"].nunique().sort_values(ascending=False).head())

# Normalización de textos
for c in ["nombre_entidad", "nombre_fondo"]:
    df[c] = (df[c]
             .astype(str)
             .str.strip()
             .str.replace(r"\s+", " ", regex=True))  

print("Cardinalidad después de limpieza:")
print(df[["nombre_entidad","nombre_fondo"]].nunique())

print("Valores únicos en nombre_entidad:", df["nombre_entidad"].unique())
print("Valores únicos en nombre_fondo:", df["nombre_fondo"].unique()[:10])

print("Conteo final nombre_entidad:")
print(df["nombre_entidad"].value_counts())
print("Conteo final nombre_fondo:")
print(df["nombre_fondo"].value_counts().head(20))

# =============================================================================
# ELIMINACIÓN DE DUPLICADOS (Complemento a tu análisis existente)
# =============================================================================

print("\n=== ANÁLISIS DE DUPLICADOS ===")
duplicados = df.duplicated().sum()
print(f"Filas duplicadas exactas: {duplicados}")

if duplicados > 0:
    print("Eliminando duplicados exactos...")
    df = df.drop_duplicates()
    print(f"Dataset después de eliminar duplicados: {len(df)} filas")
else:
    print("✓ No hay duplicados exactos")

# Buscar duplicados conceptuales (misma entidad, mismo fondo, misma fecha)
duplicados_conceptuales = df.duplicated(
    subset=['nombre_entidad', 'nombre_fondo', 'fecha']
).sum()

print(f"Duplicados conceptuales (misma entidad-fondo-fecha): {duplicados_conceptuales}")

if duplicados_conceptuales > 0:
    print("Manteniendo el primer registro de cada duplicado conceptual...")
    df = df.drop_duplicates(
        subset=['nombre_entidad', 'nombre_fondo', 'fecha'], 
        keep='first'
    )
    print(f"Dataset después de limpieza: {len(df)} filas")

# =============================================================================
# DETECCIÓN Y MANEJO DE OUTLIERS (Nuevo - Recomendación del libro)
# =============================================================================

print("\n=== ANÁLISIS DE OUTLIERS EN valor_unidad ===")

# Estadísticas robustas para detección de outliers
Q1 = df['valor_unidad'].quantile(0.25)
Q3 = df['valor_unidad'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[
    (df['valor_unidad'] < limite_inferior) | 
    (df['valor_unidad'] > limite_superior)
]

print(f"Límite inferior (outliers): {limite_inferior:.2f}")
print(f"Límite superior (outliers): {limite_superior:.2f}")
print(f"Total de outliers detectados: {len(outliers)}")
print(f"Porcentaje de outliers: {(len(outliers)/len(df)*100):.2f}%")

if len(outliers) > 0:
    print("\nMuestra de outliers:")
    print(outliers[['nombre_entidad', 'nombre_fondo', 'fecha', 'valor_unidad']].head())
    
    # Crear columna flag para outliers
    df['es_outlier'] = (
        (df['valor_unidad'] < limite_inferior) | 
        (df['valor_unidad'] > limite_superior)
    )
    
    print("✓ Columna 'es_outlier' creada para análisis posterior")
else:
    df['es_outlier'] = False
    print("✓ No se detectaron outliers significativos")

# =============================================================================
# OPTIMIZACIÓN DE TIPOS DE DATOS (Complemento a tu limpieza)
# =============================================================================

print("\nOptimizando tipos de datos...")
df['nombre_entidad'] = df['nombre_entidad'].astype('category')
df['nombre_fondo'] = df['nombre_fondo'].astype('category')
df['es_outlier'] = df['es_outlier'].astype('bool')

print("✓ Columnas convertidas a categoría para optimización")

# =============================================================================
# CREACIÓN DE VARIABLES DERIVADAS (Nuevo - Para análisis posterior)
# =============================================================================

print("\n=== CREACIÓN DE VARIABLES DERIVADAS ===")

# Extraer componentes de fecha para análisis temporal
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['trimestre'] = df['fecha'].dt.quarter

# Clasificar fondos por tipo (nueva variable categórica)
def clasificar_fondo(nombre_fondo):
    nombre = nombre_fondo.lower()
    if 'cesantia' in nombre:
        return 'Cesantías'
    elif 'pension' in nombre:
        return 'Pensiones'
    elif 'alternativo' in nombre:
        return 'Alternativo'
    else:
        return 'Otros'

df['tipo_fondo'] = df['nombre_fondo'].apply(clasificar_fondo).astype('category')

print("Variables derivadas creadas:")
print(f"  - año, mes, trimestre: para análisis temporal")
print(f"  - tipo_fondo: {df['tipo_fondo'].value_counts().to_dict()}")

# =============================================================================
# GUARDAR SUBSETS (Tu código original mantenido)
# =============================================================================

Path("data/raw").mkdir(parents=True, exist_ok=True)

def guardar_subset(df, col_filtro, valores, salida):
    if isinstance(valores, (list, tuple, set)):
        sub = df.loc[df[col_filtro].isin(valores)].copy()
    else:
        sub = df.loc[df[col_filtro].eq(valores)].copy()
    if col_filtro in sub.columns:
        sub = sub.drop(columns=[col_filtro])  
    print(sub.shape)
    sub.to_csv(salida, index=False)

guardar_subset(df_limpio, "nombre_entidad",
               "Skandia Afp - Accai S.A.",
               "data/raw/pensiones_skandia.csv")

guardar_subset(df_limpio, "nombre_entidad",
               '"Proteccion"',
               "data/raw/pensiones_proteccion.csv")

guardar_subset(df_limpio, "nombre_entidad",
               '"Porvenir"',
               "data/raw/pensiones_porvenir.csv")

guardar_subset(df_limpio, "nombre_entidad",
               '"Colfondos S.A." Y "Colfondos"',
               "data/raw/colfondos_colfondos.csv")

guardar_subset(df_limpio, "nombre_fondo",
               "Fondo de Cesantias Largo Plazo",
               "data/raw/fondo_cesantias_largo_plazo.csv")

guardar_subset(df_limpio, "nombre_fondo",
               "Fondo de Cesantias Corto Plazo",
               "data/raw/fondo_cesantias_corto_plazo.csv")

guardar_subset(df_limpio, "nombre_fondo",
               "Fondo de Pensiones Moderado",
               "data/raw/fondo_pensiones_moderado.csv")

guardar_subset(df_limpio, "nombre_fondo",
               "Fondo de Pensiones Conservador",
               "data/raw/fondo_pensiones_conservador.csv")

guardar_subset(df_limpio, "nombre_fondo",
               "Fondo de Pensiones Mayor Riesgo",
               "data/raw/fondo_pensiones_mayor_riesgo.csv")

guardar_subset(df_limpio, "nombre_fondo",
               "Fondo de Pensiones Retiro Programado",
               "data/raw/fondo_pensiones_retiro_programado.csv")

guardar_subset(df_limpio, "nombre_fondo",
               "Fondo de Pensiones Alternativo",
               "data/raw/fondo_pensiones_alternativo.csv")

# =============================================================================
# VALIDACIÓN FINAL Y EXPORTACIÓN
# =============================================================================

print("\n=== VALIDACIÓN FINAL ===")
print(f"Dimensiones finales del dataset: {df.shape}")
print(f"Tipos de datos finales:")
print(df.dtypes)
print(f"\nResumen de memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Exportar dataset limpio y listo para análisis
df.to_csv("data/processed/pensiones_limpio_final.csv", index=False, encoding='utf-8')

print("✓ Dataset limpio exportado a: data/processed/pensiones_limpio_final.csv")

# Exportar resumen de limpieza
resumen_limpieza = {
    'filas_finales': len(df),
    'columnas_finales': len(df.columns),
    'duplicados_eliminados': duplicados,
    'outliers_detectados': len(outliers),
    'memoria_mb': df.memory_usage(deep=True).sum() / 1024**2,
    'fecha_limpieza': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

pd.Series(resumen_limpieza).to_csv("data/processed/resumen_limpieza.csv")

print("✓ Resumen de limpieza exportado a: data/processed/resumen_limpieza.csv")