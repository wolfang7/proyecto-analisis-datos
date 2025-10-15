import pandas as pd
import requests
import time
import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

from scipy.stats import entropy as shannon_entropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import permutation_importance
import seaborn as sns

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

df_clean = df.drop(columns=["codigo_entidad", "codigo_patrimonio"])
df_clean.to_csv("data/raw/pensionesLimpio.csv", index=False)

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

guardar_subset(df_clean, "nombre_entidad",
               "Skandia Afp - Accai S.A.",
               "data/raw/pensiones_skandia.csv")

guardar_subset(df_clean, "nombre_entidad",
               '"Proteccion"',
               "data/raw/pensiones_proteccion.csv")

guardar_subset(df_clean, "nombre_entidad",
               '"Porvenir"',
               "data/raw/pensiones_porvenir.csv")

guardar_subset(df_clean, "nombre_entidad",
               '"Colfondos S.A." Y "Colfondos"',
               "data/raw/colfondos_colfondos.csv")

guardar_subset(df_clean, "nombre_fondo",
               "Fondo de Cesantias Largo Plazo",
               "data/raw/fondo_cesantias_largo_plazo.csv")

guardar_subset(df_clean, "nombre_fondo",
               "Fondo de Cesantias Corto Plazo",
               "data/raw/fondo_cesantias_corto_plazo.csv")

guardar_subset(df_clean, "nombre_fondo",
               "Fondo de Pensiones Moderado",
               "data/raw/fondo_pensiones_moderado.csv")

guardar_subset(df_clean, "nombre_fondo",
               "Fondo de Pensiones Conservador",
               "data/raw/fondo_pensiones_conservador.csv")

guardar_subset(df_clean, "nombre_fondo",
               "Fondo de Pensiones Mayor Riesgo",
               "data/raw/fondo_pensiones_mayor_riesgo.csv")

guardar_subset(df_clean, "nombre_fondo",
               "Fondo de Pensiones Retiro Programado",
               "data/raw/fondo_pensiones_retiro_programado.csv")

guardar_subset(df_clean, "nombre_fondo",
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

#--------------------------------------
#arboles de decision
#---------------------------------------
# Ajusta si tu columna se llama distinto
assert 'valor_unidad' in df_clean.columns, "No encuentro la columna 'valor_unidad'"

# --- CASO A: CLASIFICACIÓN (target categórica a partir de valor_unidad) ---
df_cls = df_clean.copy()
df_cls = df_cls.dropna(subset=['valor_unidad'])
df_cls['target_bin'] = pd.qcut(df_cls['valor_unidad'], q=3, labels=['bajo','medio','alto'])

# Definimos features útiles (numéricas + categóricas)
num_cols = ['valor_unidad']  # Puedes añadir más numéricas si tienes
cat_cols = ['nombre_entidad', 'nombre_fondo', 'tipo_fondo', 'nivel_riesgo']  # quita/pon según tu df

# Filtrar columnas que existan realmente
num_cols = [c for c in num_cols if c in df_cls.columns]
cat_cols = [c for c in cat_cols if c in df_cls.columns]

X = df_cls[num_cols + cat_cols].copy()
y = df_cls['target_bin'].astype('category')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
def shannon_entropy_categorical(s: pd.Series) -> float:
    counts = s.value_counts(dropna=True)
    p = counts / counts.sum()
    return float(shannon_entropy(p, base=2))

def shannon_entropy_numeric(s: pd.Series, bins: int = 20) -> float:
    s = s.dropna()
    hist, _ = np.histogram(s, bins=bins)
    p = hist / hist.sum() if hist.sum() > 0 else np.array([1.0])
    return float(shannon_entropy(p, base=2))

entropias = {}

# Categóricas
for c in cat_cols:
    entropias[c] = shannon_entropy_categorical(X_train[c].astype(str))

# Numéricas (discretizando)
for c in num_cols:
    entropias[c] = shannon_entropy_numeric(X_train[c].astype(float), bins=20)

entropias = pd.Series(entropias).sort_values(ascending=False)
print("Entropía (bits) por variable:")
print(entropias)

# Preprocesador para vectorizar X
pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', 'passthrough', num_cols)
    ],
    remainder='drop'
)

X_train_enc = pre.fit_transform(X_train)
X_test_enc  = pre.transform(X_test)

# Nombres de columnas tras One-Hot:
cat_feat_names = []
if len(cat_cols) > 0:
    cat_feat_names = list(pre.named_transformers_['cat'].get_feature_names_out(cat_cols))
num_feat_names = num_cols
feat_names = cat_feat_names + num_feat_names

# Información mutua (clasificación)
mi_scores = mutual_info_classif(X_train_enc, y_train.cat.codes, random_state=42)
mi_series = pd.Series(mi_scores, index=feat_names).sort_values(ascending=False)
print("Información mutua con la clase (mayor = más informativa):")
print(mi_series.head(20))

clf = DecisionTreeClassifier(
    criterion='entropy',  # usa entropía de Shannon
    max_depth=3,          # poco profundo para interpretar
    min_samples_leaf=50,  # ajusta según tamaño
    random_state=42
)

pipe = Pipeline(steps=[
    ('pre', pre),
    ('tree', clf)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
# Reentrenamos el árbol sobre matrices ya codificadas para poder graficar con nombres de features
clf_vis = DecisionTreeClassifier(
    criterion='entropy', max_depth=3, min_samples_leaf=50, random_state=42
)
clf_vis.fit(X_train_enc, y_train)

plt.figure(figsize=(18, 8))
plot_tree(clf_vis, feature_names=feat_names, class_names=list(y_train.cat.categories),
          filled=True, rounded=True, impurity=True)
plt.title("Árbol de decisión (criterio = entropía)")
plt.show()

# Importancias por reducción de impureza (cuidado con el sesgo en alta cardinalidad)
imp_tree = pd.Series(clf_vis.feature_importances_, index=feat_names)
print("Importancias (reducción de entropía):")
print(imp_tree.sort_values(ascending=False).head(20))

# Importancias por permutación en el pipeline completo (sobre columnas originales)
r = permutation_importance(
    pipe, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)

# Usa el índice de columnas originales, NO feat_names (expandidas)
orig_feat_names = num_cols + cat_cols

imp_perm_orig = pd.Series(r.importances_mean, index=orig_feat_names) \
                   .sort_values(ascending=False)
print("Importancias por permutación (columnas originales):")
print(imp_perm_orig.head(20))

# Guarda
imp_perm_orig.to_csv("data/reports/importancias_permutacion_original.csv")



Path("reports").mkdir(exist_ok=True)

# Top-20 MI
mi_series.head(20).to_csv("data/reports/top20_info_mutua.csv")
# al inicio del script, junto con data/...


# Importancias
imp_tree.sort_values(ascending=False).to_csv("data/reports/importancias_arbol_entropy.csv")
imp_perm_orig.to_csv("data/reports/importancias_permutacion.csv")

# Exportar figura del árbol
plt.figure(figsize=(18, 8))
plot_tree(clf_vis, feature_names=feat_names, class_names=list(y_train.cat.categories),
          filled=True, rounded=True, impurity=True)
plt.title("Árbol de decisión (criterio = entropía)")
plt.savefig("data/reports/arbol_decision_entropy.png", dpi=300, bbox_inches='tight')
plt.close()
