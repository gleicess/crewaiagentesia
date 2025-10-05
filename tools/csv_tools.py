"""
Ferramentas para processamento e análise de CSV
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import chardet
from io import StringIO
import json

from crewai_tools import tool

logger = logging.getLogger(__name__)


@tool("csv_loader")
def load_and_validate_csv(
    file_path: str, 
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Carrega e valida um arquivo CSV, detectando automaticamente encoding e separador.

    Args:
        file_path: Caminho para o arquivo CSV
        encoding: Encoding do arquivo (auto-detectado se None)
        delimiter: Separador usado no CSV (auto-detectado se None)

    Returns:
        Dict com informações de validação e os dados carregados
    """
    try:
        # Auto-detectar encoding se não fornecido
        if encoding is None:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']

        # Tentar carregar o CSV
        df = pd.read_csv(file_path, encoding=encoding, sep=delimiter)

        # Auto-detectar separador se necessário
        if delimiter is None and df.shape[1] == 1:
            # Tentar diferentes separadores
            separators = [';', '\t', '|', ':']
            for sep in separators:
                try:
                    test_df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    if test_df.shape[1] > df.shape[1]:
                        df = test_df
                        delimiter = sep
                        break
                except:
                    continue

        validation_result = {
            "success": True,
            "file_path": file_path,
            "encoding": encoding,
            "delimiter": delimiter or ",",
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "data": df.to_dict('records')[:1000],  # Primeiros 1000 registros
            "data_types_analysis": analyze_column_types(df),
            "quality_report": generate_quality_report(df)
        }

        return validation_result

    except Exception as e:
        logger.error(f"Erro ao carregar CSV: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }


@tool("analyze_column_types")
def analyze_column_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analisa e identifica os tipos de cada coluna do DataFrame.

    Args:
        df: DataFrame do pandas

    Returns:
        Dict com análise detalhada de cada coluna
    """
    analysis = {}

    for col in df.columns:
        col_analysis = {
            "pandas_dtype": str(df[col].dtype),
            "null_count": df[col].isnull().sum(),
            "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
            "unique_values": df[col].nunique(),
            "unique_percentage": (df[col].nunique() / len(df)) * 100
        }

        # Identificar tipo semântico
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            col_analysis["semantic_type"] = "numerical"
            col_analysis["min_value"] = df[col].min()
            col_analysis["max_value"] = df[col].max()
            col_analysis["mean"] = df[col].mean()
            col_analysis["std"] = df[col].std()

        elif df[col].dtype == 'object':
            # Tentar detectar datas
            try:
                pd.to_datetime(df[col].dropna().head(100), infer_datetime_format=True)
                col_analysis["semantic_type"] = "datetime"
            except:
                # Verificar se é categórico
                if col_analysis["unique_percentage"] < 5:  # Menos de 5% de valores únicos
                    col_analysis["semantic_type"] = "categorical"
                    col_analysis["categories"] = df[col].value_counts().head(10).to_dict()
                else:
                    col_analysis["semantic_type"] = "text"
                    col_analysis["avg_length"] = df[col].astype(str).str.len().mean()

        elif df[col].dtype == 'bool':
            col_analysis["semantic_type"] = "boolean"
            col_analysis["true_count"] = df[col].sum()
            col_analysis["false_count"] = len(df) - df[col].sum()

        else:
            col_analysis["semantic_type"] = "other"

        analysis[col] = col_analysis

    return analysis


@tool("quality_report")
def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Gera um relatório de qualidade dos dados.

    Args:
        df: DataFrame do pandas

    Returns:
        Dict com relatório de qualidade
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100,
        "completely_null_rows": df.isnull().all(axis=1).sum(),
        "completely_null_columns": df.isnull().all().sum(),
        "columns_with_nulls": df.isnull().any().sum(),
        "total_null_values": df.isnull().sum().sum(),
        "null_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
    }

    # Análise de outliers para colunas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers_analysis = {}

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        outliers_analysis[col] = {
            "outlier_count": len(outliers),
            "outlier_percentage": (len(outliers) / len(df)) * 100,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    report["outliers_analysis"] = outliers_analysis

    # Recomendações de limpeza
    recommendations = []

    if report["duplicate_percentage"] > 1:
        recommendations.append("Considere remover linhas duplicadas")

    if report["null_percentage"] > 10:
        recommendations.append("Alto percentual de valores nulos - considere estratégias de imputação")

    if report["completely_null_rows"] > 0:
        recommendations.append("Remover linhas completamente vazias")

    if report["completely_null_columns"] > 0:
        recommendations.append("Remover colunas completamente vazias")

    report["recommendations"] = recommendations

    return report


@tool("data_sampler")
def sample_data_for_analysis(df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    """
    Cria uma amostra representativa dos dados para análise.

    Args:
        df: DataFrame original
        sample_size: Tamanho da amostra

    Returns:
        DataFrame com amostra dos dados
    """
    if len(df) <= sample_size:
        return df

    # Amostragem estratificada se possível
    try:
        # Tentar identificar coluna categórica para estratificação
        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(categorical_cols) > 0:
            strat_col = categorical_cols[0]
            # Amostragem estratificada
            sample = df.groupby(strat_col, group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size // df[strat_col].nunique()))
            )
            return sample.sample(n=min(len(sample), sample_size))
        else:
            # Amostragem aleatória simples
            return df.sample(n=sample_size)

    except Exception:
        # Fallback para amostragem aleatória
        return df.sample(n=sample_size)


@tool("csv_converter")
def convert_data_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Converte tipos de dados das colunas conforme especificado.

    Args:
        df: DataFrame original
        type_mapping: Dict mapeando coluna -> novo tipo

    Returns:
        DataFrame com tipos convertidos
    """
    df_converted = df.copy()

    for col, new_type in type_mapping.items():
        if col not in df.columns:
            continue

        try:
            if new_type == 'datetime':
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
            elif new_type == 'category':
                df_converted[col] = df_converted[col].astype('category')
            elif new_type in ['int', 'int64']:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype('int64')
            elif new_type in ['float', 'float64']:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            elif new_type == 'string':
                df_converted[col] = df_converted[col].astype(str)
            elif new_type == 'boolean':
                df_converted[col] = df_converted[col].astype(bool)

        except Exception as e:
            logger.warning(f"Não foi possível converter coluna {col} para {new_type}: {e}")

    return df_converted
