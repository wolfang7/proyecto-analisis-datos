"""
Pipeline generalizado para análisis reproducible de datos en Python.

Se inspira en el flujo implementado para el dataset de pensiones (`pensiones.py`)
pero ofrece parámetros configurables para aplicar las mismas etapas a cualquier
fuente tabular. Está alineado con la pregunta de investigación:

    ¿Cómo diseñar y ejecutar un flujo reproducible de análisis de datos en Python,
    desde la limpieza hasta la comunicación de hallazgos, aplicado a un conjunto de datos público?

Objetivos alcanzados con este módulo:
1. Ingesta configurable (API Socrata o archivos locales).
2. Limpieza y control de calidad parametrizable.
3. Detección de duplicados y outliers adaptable a múltiples columnas.
4. Enriquecimiento y exportación estandarizada de artefactos de datos.
5. Modelo exploratorio opcional respaldado por scikit-learn y SciPy.

El módulo mantiene las dependencias utilizadas en `pensiones.py`, evitando
nuevas librerías. Para ejecutar, crea una instancia de `PipelineConfig` y llama
a `run_pipeline(config)`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.stats import entropy as shannon_entropy
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ---------------------------------------------------------------------------
# Configuración y reglas de negocio
# ---------------------------------------------------------------------------

@dataclass
class SubsetRule:
    """Define un subconjunto que debe exportarse a partir de un DataFrame limpio."""

    column: str
    values: Sequence[str]
    filename: str
    match: str = "exact"  # "exact" | "contains"


@dataclass
class PipelineConfig:
    """
    Parametrización del pipeline.

    Atributos más relevantes:
    - source_type: "socrata" o "csv".
    - source: URL base (para API) o ruta de archivo local.
    - resource_id: requerido para Socrata.
    - datetime_columns / numeric_columns / text_columns: definen limpieza.
    - duplicate_keys: combinaciones de columnas para deduplicar.
    - outlier_columns: columnas numéricas a evaluar con IQR.
    - date_feature_columns: columnas datetime a descomponer (año, mes, trimestre).
    - lookup_pairs: pares (columna_nombre, columna_codigo) para generar catálogos.
    - subset_rules: subconjuntos que se exportarán en data/raw.
    - target_column: si se define, permite crear un árbol de decisión exploratorio.
    """

    dataset_name: str
    source_type: str
    source: str
    resource_id: Optional[str] = None
    pagination_limit: int = 50_000
    pagination_params: Dict[str, str] = field(default_factory=dict)
    datetime_columns: Sequence[str] = field(default_factory=tuple)
    numeric_columns: Sequence[str] = field(default_factory=tuple)
    numeric_clean_regex: str = r"[^\d\-,\.]"
    decimal_separator: Optional[str] = ","
    text_columns: Sequence[str] = field(default_factory=tuple)
    duplicate_keys: Sequence[Sequence[str]] = field(default_factory=tuple)
    outlier_columns: Sequence[str] = field(default_factory=tuple)
    date_feature_columns: Sequence[str] = field(default_factory=tuple)
    text_as_category: Sequence[str] = field(default_factory=tuple)
    lookup_pairs: Sequence[Tuple[str, str]] = field(default_factory=tuple)
    subset_rules: Sequence[SubsetRule] = field(default_factory=tuple)
    target_column: Optional[str] = None
    classification_bins: int = 3
    num_model_columns: Sequence[str] = field(default_factory=tuple)
    cat_model_columns: Sequence[str] = field(default_factory=tuple)
    output_root: Path = field(default_factory=lambda: Path("data"))


# ---------------------------------------------------------------------------
# Utilidades de directorios y estilos
# ---------------------------------------------------------------------------

def prepare_output_directories(output_root: Path) -> Dict[str, Path]:
    """
    Crea la estructura estándar de carpetas (raw, processed, reports).

    Retorna:
        dict con rutas -> {"raw": Path, "processed": Path, "reports": Path}
    """
    directories = {
        "raw": output_root / "raw",
        "processed": output_root / "processed",
        "reports": output_root / "reports",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.max_columns", None)
    sns.set_theme(style="whitegrid")
    return directories


# ---------------------------------------------------------------------------
# Ingesta de datos
# ---------------------------------------------------------------------------

def fetch_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Descarga o carga el dataset según la configuración indicada."""
    if config.source_type.lower() == "csv":
        df = pd.read_csv(config.source)
        print(f"Archivo CSV cargado: {len(df)} filas.")
        return df

    if config.source_type.lower() != "socrata":
        raise ValueError(f"source_type no soportado: {config.source_type}")

    if not config.resource_id:
        raise ValueError("Para source_type='socrata' se requiere resource_id.")

    base = config.source.rstrip("/")
    url = f"{base}/resource/{config.resource_id}.json"

    try:
        total = requests.get(f"{url}?$select=count(*)", timeout=60).json()[0]["count"]
        print(f"Total reportado por el endpoint: {total}")
    except Exception:
        print("No se pudo obtener el total reportado por la API.")

    pages: List[pd.DataFrame] = []
    offset = 0

    while True:
        params = {"$limit": config.pagination_limit, "$offset": offset, **config.pagination_params}
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        chunk = response.json()
        if not chunk:
            break
        pages.append(pd.DataFrame(chunk))
        offset += config.pagination_limit
        print(f"Descargadas: {offset} filas…")
        time.sleep(0.3)

    if not pages:
        return pd.DataFrame()

    return pd.concat(pages, ignore_index=True)


# ---------------------------------------------------------------------------
# Limpieza y control de calidad
# ---------------------------------------------------------------------------

def coerce_types(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Aplica conversiones de tipo basadas en la configuración."""
    df = df.copy()

    for col in config.datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in config.numeric_columns:
        if col in df.columns:
            series = df[col].astype(str)
            if config.numeric_clean_regex:
                series = series.str.replace(config.numeric_clean_regex, "", regex=True)
            if config.decimal_separator and config.decimal_separator != ".":
                series = series.str.replace(config.decimal_separator, ".", regex=False)
            df[col] = pd.to_numeric(series, errors="coerce")

    return df


def compute_quality_summary(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, object]:
    """Calcula métricas descriptivas para evaluar calidad de datos."""
    null_pct = df.isna().mean().sort_values(ascending=False).mul(100).round(2)
    cardinalidad = df.nunique(dropna=True).sort_values(ascending=False)

    relaciones = {}
    for nombre_col, codigo_col in config.lookup_pairs:
        if nombre_col in df.columns and codigo_col in df.columns:
            relaciones[f"{codigo_col}->{nombre_col}"] = (
                df.groupby(codigo_col)[nombre_col]
                .nunique()
                .sort_values(ascending=False)
                .head()
            )

    return {
        "nulos_pct": null_pct,
        "cardinalidad": cardinalidad,
        "relaciones": relaciones,
    }


def export_reference_maps(df: pd.DataFrame, config: PipelineConfig, directories: Dict[str, Path]) -> Dict[str, Dict[str, str]]:
    """Exporta catálogos clave (p. ej., nombre vs. código) según lookup_pairs."""
    raw_dir = directories["raw"]
    mappings: Dict[str, Dict[str, str]] = {}

    for nombre_col, codigo_col in config.lookup_pairs:
        if nombre_col not in df.columns or codigo_col not in df.columns:
            continue

        catalogo = (
            df[[nombre_col, codigo_col]]
            .drop_duplicates()
            .dropna(subset=[nombre_col, codigo_col])
            .set_index(nombre_col)[codigo_col]
            .to_dict()
        )
        mappings[f"{nombre_col}->{codigo_col}"] = catalogo
        nombre_archivo = f"{nombre_col}_{codigo_col}.csv".replace(" ", "_").lower()
        pd.DataFrame(list(catalogo.items()), columns=[nombre_col, codigo_col]).to_csv(
            raw_dir / nombre_archivo, index=False
        )

    return mappings


def normalize_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Homogeneiza espacios y capitalización en columnas de texto."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
    return df


# ---------------------------------------------------------------------------
# Duplicados, outliers y enriquecimiento
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame, duplicate_keys: Sequence[Sequence[str]]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Elimina duplicados exactos y por claves definidas."""
    df = df.copy()
    metrics: Dict[str, int] = {}

    exact = int(df.duplicated().sum())
    metrics["duplicados_exactos"] = exact
    if exact:
        df = df.drop_duplicates()

    for key_group in duplicate_keys:
        key_tuple = tuple(key_group)
        if not set(key_tuple).issubset(df.columns):
            continue
        duplicados = int(df.duplicated(subset=list(key_tuple)).sum())
        metrics[f"duplicados_{'_'.join(key_tuple)}"] = duplicados
        if duplicados:
            df = df.drop_duplicates(subset=list(key_tuple), keep="first")

    return df, metrics


def detect_outliers(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Aplica detección de outliers basada en IQR para cada columna numérica."""
    df = df.copy()
    report: Dict[str, Dict[str, float]] = {}

    for col in columns:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)

        df[f"{col}_es_outlier"] = mask
        report[col] = {
            "limite_inferior": float(lower),
            "limite_superior": float(upper),
            "total": int(mask.sum()),
            "porcentaje": float(mask.mean() * 100),
        }

    return df, report


def add_derived_columns(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Genera variables derivadas genéricas (fecha y tipificación de texto)."""
    df = df.copy()

    for col in config.text_as_category:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in config.date_feature_columns:
        if col in df.columns:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_quarter"] = df[col].dt.quarter

    return df


# ---------------------------------------------------------------------------
# Exportación de artefactos
# ---------------------------------------------------------------------------

def export_primary_outputs(
    df_processed: pd.DataFrame,
    df_clean: pd.DataFrame,
    directories: Dict[str, Path],
    dedup_metrics: Dict[str, int],
    outlier_metrics: Dict[str, Dict[str, float]],
    config: PipelineConfig,
) -> None:
    """Guarda datasets principales y un resumen del pipeline ejecutado."""
    raw_dir = directories["raw"]
    processed_dir = directories["processed"]

    df_clean.to_csv(raw_dir / f"{config.dataset_name}_clean.csv", index=False)
    df_processed.to_csv(processed_dir / f"{config.dataset_name}_processed.csv", index=False, encoding="utf-8")

    resumen = {
        "filas_finales": len(df_processed),
        "columnas_finales": len(df_processed.columns),
        "memoria_mb": float(df_processed.memory_usage(deep=True).sum() / 1024**2),
        "fecha_limpieza": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        **dedup_metrics,
    }

    for col, stats in outlier_metrics.items():
        resumen[f"outliers_{col}_total"] = stats["total"]
        resumen[f"outliers_{col}_pct"] = stats["porcentaje"]

    pd.Series(resumen).to_csv(processed_dir / f"{config.dataset_name}_resumen.csv")


def export_subsets(df: pd.DataFrame, rules: Sequence[SubsetRule], directories: Dict[str, Path]) -> None:
    """Genera subconjuntos temáticos definidos por el usuario."""
    raw_dir = directories["raw"]

    for rule in rules:
        if rule.column not in df.columns:
            continue

        values = list(rule.values)
        if rule.match == "contains":
            mask = pd.Series(False, index=df.index)
            for value in values:
                mask |= df[rule.column].astype(str).str.contains(value, case=False, na=False)
        else:
            mask = df[rule.column].isin(values)

        subset = df.loc[mask].drop(columns=[rule.column], errors="ignore")
        subset.to_csv(raw_dir / rule.filename, index=False)
        print(f"Subconjunto '{rule.filename}' generado con {len(subset)} filas.")


# ---------------------------------------------------------------------------
# Modelo exploratorio opcional
# ---------------------------------------------------------------------------

def create_one_hot_encoder() -> OneHotEncoder:
    """Instancia un OneHotEncoder compatible con distintas versiones de scikit-learn."""
    params = inspect.signature(OneHotEncoder).parameters
    kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in params:
        kwargs["sparse_output"] = False
    elif "sparse" in params:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def build_decision_tree(df: pd.DataFrame, config: PipelineConfig, directories: Dict[str, Path]) -> None:
    """Entrena un árbol de decisión simple si se dispone del target y columnas de entrada."""
    if not config.target_column:
        print("Sin target_column definido: se omite el modelo exploratorio.")
        return
    if config.target_column not in df.columns:
        print(f"Target '{config.target_column}' no encontrado: se omite el modelo exploratorio.")
        return

    df_model = df.dropna(subset=[config.target_column]).copy()
    if df_model.empty:
        print("No hay datos suficientes para entrenar el árbol de decisión.")
        return

    target = df_model[config.target_column]
    if pd.api.types.is_numeric_dtype(target) and config.classification_bins > 1:
        df_model["__target_bin"] = pd.qcut(
            target,
            q=config.classification_bins,
            labels=[f"bin_{i}" for i in range(config.classification_bins)],
        )
    else:
        df_model["__target_bin"] = target.astype("category")

    num_cols = [col for col in config.num_model_columns if col in df_model.columns]
    cat_cols = [col for col in config.cat_model_columns if col in df_model.columns]
    if not (num_cols or cat_cols):
        print("Sin columnas numéricas o categóricas definidas para el modelo.")
        return

    X = df_model[num_cols + cat_cols].copy()
    y = df_model["__target_bin"].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    def entropy_for_series(series: pd.Series, bins: int = 20) -> float:
        if series.dtype.kind in {"O", "b"}:
            counts = series.value_counts(dropna=True)
            probs = counts / counts.sum()
            return float(shannon_entropy(probs, base=2))
        clean = series.dropna().astype(float)
        hist, _ = np.histogram(clean, bins=bins)
        probs = hist / hist.sum() if hist.sum() else np.array([1.0])
        return float(shannon_entropy(probs, base=2))

    entropias = {col: entropy_for_series(X_train[col]) for col in X_train.columns}
    pd.Series(entropias).sort_values(ascending=False).to_csv(directories["reports"] / "entropia_variables.csv")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", create_one_hot_encoder(), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    cat_feature_names = (
        list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
        if cat_cols
        else []
    )
    feature_names = cat_feature_names + num_cols

    mi_scores = mutual_info_classif(X_train_enc, y_train.cat.codes, random_state=42)
    pd.Series(mi_scores, index=feature_names).sort_values(ascending=False).head(20).to_csv(
        directories["reports"] / "top20_info_mutua.csv"
    )

    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,
        min_samples_leaf=50,
        random_state=42,
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("tree", clf),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    clf_vis = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,
        min_samples_leaf=50,
        random_state=42,
    )
    clf_vis.fit(X_train_enc, y_train)

    plt.figure(figsize=(18, 8))
    plot_tree(
        clf_vis,
        feature_names=feature_names,
        class_names=list(y_train.cat.categories),
        filled=True,
        rounded=True,
        impurity=True,
    )
    plt.title("Árbol de decisión (criterio = entropía)")
    plt.savefig(directories["reports"] / "arbol_decision_entropy.png", dpi=300, bbox_inches="tight")
    plt.close()

    importancias_arbol = pd.Series(clf_vis.feature_importances_, index=feature_names)
    importancias_arbol.sort_values(ascending=False).to_csv(directories["reports"] / "importancias_arbol_entropy.csv")

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    perm_series = pd.Series(perm.importances_mean, index=num_cols + cat_cols).sort_values(ascending=False)
    perm_series.to_csv(directories["reports"] / "importancias_permutacion.csv")


# ---------------------------------------------------------------------------
# Utilidades de resumen
# ---------------------------------------------------------------------------

def print_summary(title: str, items: Dict[str, object]) -> None:
    """Imprime un bloque de información resumida en consola."""
    print(f"\n=== {title} ===")
    for key, value in items.items():
        print(f"- {key}: {value}")


# ---------------------------------------------------------------------------
# Función orquestadora
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> None:
    """Ejecuta todas las etapas del pipeline de acuerdo con el `PipelineConfig`."""
    print(f"Iniciando pipeline para: {config.dataset_name}")
    directories = prepare_output_directories(config.output_root)

    df_raw = fetch_dataset(config)
    print(f"Filas descargadas/cargadas: {len(df_raw)}")
    if df_raw.empty:
        print("No se obtuvo información. Se detiene el pipeline.")
        return

    df = coerce_types(df_raw, config)
    df = normalize_text_columns(df, config.text_columns)

    quality = compute_quality_summary(df, config)
    reference_maps = export_reference_maps(df, config, directories)

    df, dedup_metrics = remove_duplicates(df, config.duplicate_keys)
    df, outlier_metrics = detect_outliers(df, config.outlier_columns)
    df = add_derived_columns(df, config)

    columns_to_drop = {codigo for _, codigo in config.lookup_pairs if codigo in df.columns}
    df_clean = df.drop(columns=list(columns_to_drop), errors="ignore").copy()

    export_primary_outputs(df, df_clean, directories, dedup_metrics, outlier_metrics, config)
    export_subsets(df_clean, config.subset_rules, directories)
    build_decision_tree(df_clean, config, directories)

    print_summary("Calidad de datos", {
        "nulos_top": quality["nulos_pct"].head(),
        "cardinalidad_top": quality["cardinalidad"].head(),
        "relaciones": quality["relaciones"],
    })
    print_summary("Diccionarios exportados", {k: len(v) for k, v in reference_maps.items()})
    print_summary("Duplicados", dedup_metrics)
    print_summary("Outliers", {col: stats for col, stats in outlier_metrics.items()})

    print("\nPipeline finalizado. Revisar 'data/raw', 'data/processed' y 'data/reports'.")


# ---------------------------------------------------------------------------
# Ejemplo de uso (personalizar según dataset objetivo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_config = PipelineConfig(
        dataset_name="dataset_demo",
        source_type="socrata",
        source="https://www.datos.gov.co",
        resource_id="uawh-cjvi",  # Reemplazar con el recurso objetivo
        datetime_columns=("fecha",),
        numeric_columns=("valor_unidad",),
        text_columns=("nombre_entidad", "nombre_fondo"),
        duplicate_keys=(("nombre_entidad", "nombre_fondo", "fecha"),),
        outlier_columns=("valor_unidad",),
        date_feature_columns=("fecha",),
        text_as_category=("nombre_entidad", "nombre_fondo"),
        lookup_pairs=(("nombre_entidad", "codigo_entidad"), ("nombre_fondo", "codigo_patrimonio")),
        subset_rules=(
            SubsetRule("nombre_entidad", ["Colfondos"], "colfondos.csv", "contains"),
        ),
        target_column="valor_unidad",
        classification_bins=3,
        num_model_columns=("valor_unidad",),
        cat_model_columns=("nombre_entidad", "nombre_fondo"),
        output_root=Path("data"),
    )

    # La siguiente línea ejecuta todo el pipeline con la configuración de ejemplo.
    run_pipeline(example_config)
