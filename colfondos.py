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
df.to_csv("data/raw/pensionesLimpio.csv", index=False)
# % de nulos por columna
nulls = df.isna().mean().sort_values(ascending=False).mul(100).round(2)
print(nulls)
cardinalidad = df.nunique(dropna=True).sort_values(ascending=False)
print(cardinalidad)

# valores únicos (muestra)
df["nombre_entidad"].dropna().unique()[:10]
df["nombre_fondo"].dropna().unique()[:10]
df["nombre_entidad"].value_counts(dropna=False).head(10)
df["nombre_fondo"].value_counts(dropna=False).head(20)

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
print(df.groupby("codigo_entidad")["nombre_entidad"].nunique().sort_values(ascending=False).head())
print(df.groupby("codigo_patrimonio")["nombre_fondo"].nunique().sort_values(ascending=False).head())

# y al revés: cuántos códigos por nombre
print(df.groupby("nombre_entidad")["codigo_entidad"].nunique().sort_values(ascending=False).head())
print(df.groupby("nombre_fondo")["codigo_patrimonio"].nunique().sort_values(ascending=False).head())


for c in ["nombre_entidad", "nombre_fondo"]:
    df[c] = (df[c]
             .astype(str)
             .str.strip()
             .str.replace(r"\s+", " ", regex=True))  


print(df[["nombre_entidad","nombre_fondo"]].nunique())

print("Valores únicos en nombre_entidad:", df["nombre_entidad"].unique())
print("Valores únicos en nombre_fondo:", df["nombre_fondo"].unique()[:10])

print(df["nombre_entidad"].value_counts())
print(df["nombre_fondo"].value_counts().head(20))


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























