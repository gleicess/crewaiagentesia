"""
Funções de validação para o sistema de agentes CSV
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import re
from pathlib import Path
import mimetypes
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exceção customizada para erros de validação"""
    pass

def validate_file_format(file_path: str, allowed_formats: List[str] = None) -> Dict[str, Any]:
    """
    Valida se o formato do arquivo é suportado

    Args:
        file_path: Caminho para o arquivo
        allowed_formats: Lista de formatos permitidos

    Returns:
        Dict com resultado da validação
    """
    if allowed_formats is None:
        allowed_formats = ['.csv', '.xlsx', '.xls', '.tsv', '.txt']

    try:
        file_path = Path(file_path)

        if not file_path.exists():
            return {
                "valid": False,
                "error": f"Arquivo não encontrado: {file_path}",
                "error_code": "FILE_NOT_FOUND"
            }

        file_extension = file_path.suffix.lower()

        if file_extension not in allowed_formats:
            return {
                "valid": False,
                "error": f"Formato {file_extension} não suportado. Formatos aceitos: {', '.join(allowed_formats)}",
                "error_code": "UNSUPPORTED_FORMAT",
                "detected_format": file_extension,
                "allowed_formats": allowed_formats
            }

        # Verificar MIME type se possível
        mime_type, _ = mimetypes.guess_type(str(file_path))

        return {
            "valid": True,
            "file_extension": file_extension,
            "mime_type": mime_type,
            "file_size_bytes": file_path.stat().st_size
        }

    except Exception as e:
        return {
            "valid": False,
            "error": f"Erro na validação do formato: {str(e)}",
            "error_code": "VALIDATION_ERROR"
        }


def validate_file_size(file_path: str, max_size_mb: float = 100.0, 
                      min_size_bytes: int = 1) -> Dict[str, Any]:
    """
    Valida se o tamanho do arquivo está dentro dos limites

    Args:
        file_path: Caminho para o arquivo
        max_size_mb: Tamanho máximo em MB
        min_size_bytes: Tamanho mínimo em bytes

    Returns:
        Dict com resultado da validação
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            return {
                "valid": False,
                "error": f"Arquivo não encontrado: {file_path}",
                "error_code": "FILE_NOT_FOUND"
            }

        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        if file_size_bytes < min_size_bytes:
            return {
                "valid": False,
                "error": f"Arquivo muito pequeno: {file_size_bytes} bytes (mínimo: {min_size_bytes} bytes)",
                "error_code": "FILE_TOO_SMALL",
                "file_size_bytes": file_size_bytes,
                "min_size_bytes": min_size_bytes
            }

        if file_size_mb > max_size_mb:
            return {
                "valid": False,
                "error": f"Arquivo muito grande: {file_size_mb:.2f} MB (máximo: {max_size_mb} MB)",
                "error_code": "FILE_TOO_LARGE",
                "file_size_mb": file_size_mb,
                "max_size_mb": max_size_mb
            }

        return {
            "valid": True,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": file_size_mb,
            "within_limits": True
        }

    except Exception as e:
        return {
            "valid": False,
            "error": f"Erro na validação do tamanho: {str(e)}",
            "error_code": "VALIDATION_ERROR"
        }


def validate_dataframe_structure(df: pd.DataFrame, min_rows: int = 1, 
                               min_columns: int = 1, max_columns: int = 1000) -> Dict[str, Any]:
    """
    Valida a estrutura básica do DataFrame

    Args:
        df: DataFrame para validar
        min_rows: Número mínimo de linhas
        min_columns: Número mínimo de colunas
        max_columns: Número máximo de colunas

    Returns:
        Dict com resultado da validação
    """
    try:
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "shape": df.shape,
            "columns": list(df.columns)
        }

        rows, cols = df.shape

        # Validar número de linhas
        if rows < min_rows:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "type": "INSUFFICIENT_ROWS",
                "message": f"DataFrame tem apenas {rows} linhas (mínimo: {min_rows})",
                "current_value": rows,
                "required_minimum": min_rows
            })

        # Validar número de colunas
        if cols < min_columns:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "type": "INSUFFICIENT_COLUMNS",
                "message": f"DataFrame tem apenas {cols} colunas (mínimo: {min_columns})",
                "current_value": cols,
                "required_minimum": min_columns
            })

        if cols > max_columns:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "type": "TOO_MANY_COLUMNS",
                "message": f"DataFrame tem {cols} colunas (máximo: {max_columns})",
                "current_value": cols,
                "allowed_maximum": max_columns
            })

        # Verificar se há colunas vazias
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            validation_result["warnings"].append({
                "type": "EMPTY_COLUMNS",
                "message": f"Colunas completamente vazias: {empty_columns}",
                "empty_columns": empty_columns
            })

        # Verificar se há linhas vazias
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            validation_result["warnings"].append({
                "type": "EMPTY_ROWS",
                "message": f"{empty_rows} linhas completamente vazias",
                "empty_rows_count": empty_rows
            })

        # Verificar nomes de colunas duplicados
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            validation_result["errors"].append({
                "type": "DUPLICATE_COLUMNS",
                "message": f"Nomes de colunas duplicados: {duplicate_columns}",
                "duplicate_columns": duplicate_columns
            })
            validation_result["valid"] = False

        return validation_result

    except Exception as e:
        return {
            "valid": False,
            "error": f"Erro na validação da estrutura: {str(e)}",
            "error_code": "VALIDATION_ERROR"
        }


def validate_column_names(df: pd.DataFrame, strict_mode: bool = False) -> Dict[str, Any]:
    """
    Valida nomes de colunas

    Args:
        df: DataFrame para validar
        strict_mode: Se True, aplica regras mais rigorosas

    Returns:
        Dict com resultado da validação
    """
    try:
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }

        for i, col in enumerate(df.columns):
            col_str = str(col)

            # Verificar se é string vazia ou apenas espaços
            if not col_str.strip():
                validation_result["errors"].append({
                    "type": "EMPTY_COLUMN_NAME",
                    "message": f"Coluna {i} tem nome vazio",
                    "column_index": i,
                    "current_name": col
                })
                validation_result["valid"] = False

            # Verificar caracteres especiais problemáticos
            problematic_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            if any(char in col_str for char in problematic_chars):
                validation_result["warnings"].append({
                    "type": "PROBLEMATIC_CHARACTERS",
                    "message": f"Coluna '{col}' contém caracteres problemáticos",
                    "column_name": col,
                    "problematic_chars": [char for char in problematic_chars if char in col_str]
                })

            # Verificar se começa com número (problemático para algumas operações)
            if col_str and col_str[0].isdigit():
                validation_result["warnings"].append({
                    "type": "STARTS_WITH_NUMBER",
                    "message": f"Coluna '{col}' começa com número",
                    "column_name": col
                })

            # Modo estrito: verificar convenções de nomenclatura
            if strict_mode:
                # Verificar se contém espaços
                if ' ' in col_str:
                    validation_result["suggestions"].append({
                        "type": "CONTAINS_SPACES",
                        "message": f"Coluna '{col}' contém espaços (sugestão: usar underscore)",
                        "column_name": col,
                        "suggested_name": col_str.replace(' ', '_')
                    })

                # Verificar se está em CamelCase ou snake_case
                if not (col_str.islower() or col_str.isupper() or '_' in col_str):
                    if any(c.isupper() for c in col_str[1:]):
                        validation_result["suggestions"].append({
                            "type": "MIXED_CASE",
                            "message": f"Coluna '{col}' usa CamelCase (sugestão: snake_case)",
                            "column_name": col,
                            "suggested_name": re.sub(r'(?<!^)(?=[A-Z])', '_', col_str).lower()
                        })

        return validation_result

    except Exception as e:
        return {
            "valid": False,
            "error": f"Erro na validação dos nomes de colunas: {str(e)}",
            "error_code": "VALIDATION_ERROR"
        }


def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Valida tipos de dados das colunas

    Args:
        df: DataFrame para validar
        expected_types: Dict mapeando coluna -> tipo esperado

    Returns:
        Dict com resultado da validação
    """
    try:
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "type_analysis": {}
        }

        for col in df.columns:
            col_analysis = {
                "current_type": str(df[col].dtype),
                "null_count": df[col].isnull().sum(),
                "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "unique_count": df[col].nunique(),
                "unique_percentage": (df[col].nunique() / len(df)) * 100
            }

            # Verificar se o tipo esperado coincide com o atual
            if expected_types and col in expected_types:
                expected_type = expected_types[col]
                current_type = str(df[col].dtype)

                if expected_type != current_type:
                    validation_result["warnings"].append({
                        "type": "TYPE_MISMATCH",
                        "message": f"Coluna '{col}': esperado {expected_type}, encontrado {current_type}",
                        "column_name": col,
                        "expected_type": expected_type,
                        "current_type": current_type
                    })

            # Verificar inconsistências específicas

            # Coluna numérica com muitos valores únicos (pode indicar ID)
            if pd.api.types.is_numeric_dtype(df[col]):
                if col_analysis["unique_percentage"] > 95:
                    validation_result["warnings"].append({
                        "type": "POTENTIAL_ID_COLUMN",
                        "message": f"Coluna '{col}' numérica com {col_analysis['unique_percentage']:.1f}% valores únicos (pode ser ID)",
                        "column_name": col,
                        "unique_percentage": col_analysis["unique_percentage"]
                    })

                # Verificar valores infinitos
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    validation_result["errors"].append({
                        "type": "INFINITE_VALUES",
                        "message": f"Coluna '{col}' contém {inf_count} valores infinitos",
                        "column_name": col,
                        "infinite_count": inf_count
                    })
                    validation_result["valid"] = False

            # Coluna categórica com muitas categorias
            elif df[col].dtype == 'object':
                if col_analysis["unique_percentage"] > 50 and col_analysis["unique_count"] > 100:
                    validation_result["warnings"].append({
                        "type": "HIGH_CARDINALITY_CATEGORICAL",
                        "message": f"Coluna '{col}' categórica com {col_analysis['unique_count']} valores únicos",
                        "column_name": col,
                        "unique_count": col_analysis["unique_count"]
                    })

                # Verificar se pode ser convertida para datetime
                if col_analysis["unique_count"] > 10:  # Pelo menos algumas variações
                    sample_values = df[col].dropna().head(100)
                    try:
                        pd.to_datetime(sample_values, infer_datetime_format=True)
                        validation_result["suggestions"] = validation_result.get("suggestions", [])
                        validation_result["suggestions"].append({
                            "type": "POTENTIAL_DATETIME",
                            "message": f"Coluna '{col}' pode ser convertida para datetime",
                            "column_name": col
                        })
                    except:
                        pass

            validation_result["type_analysis"][col] = col_analysis

        return validation_result

    except Exception as e:
        return {
            "valid": False,
            "error": f"Erro na validação dos tipos de dados: {str(e)}",
            "error_code": "VALIDATION_ERROR"
        }


def validate_data_quality(df: pd.DataFrame, quality_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Valida qualidade geral dos dados

    Args:
        df: DataFrame para validar
        quality_thresholds: Thresholds para métricas de qualidade

    Returns:
        Dict com resultado da validação
    """
    if quality_thresholds is None:
        quality_thresholds = {
            "max_null_percentage": 50.0,
            "max_duplicate_percentage": 25.0,
            "min_unique_percentage": 0.1,
            "max_outlier_percentage": 10.0
        }

    try:
        validation_result = {
            "valid": True,
            "quality_score": 0.0,
            "issues": [],
            "metrics": {}
        }

        # Calcular métricas de qualidade
        total_cells = len(df) * len(df.columns)
        null_count = df.isnull().sum().sum()
        null_percentage = (null_count / total_cells) * 100

        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100

        validation_result["metrics"] = {
            "null_percentage": null_percentage,
            "duplicate_percentage": duplicate_percentage,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }

        # Verificar thresholds
        if null_percentage > quality_thresholds["max_null_percentage"]:
            validation_result["issues"].append({
                "type": "HIGH_NULL_PERCENTAGE",
                "severity": "high",
                "message": f"Alto percentual de valores ausentes: {null_percentage:.1f}%",
                "current_value": null_percentage,
                "threshold": quality_thresholds["max_null_percentage"]
            })
            validation_result["valid"] = False

        if duplicate_percentage > quality_thresholds["max_duplicate_percentage"]:
            validation_result["issues"].append({
                "type": "HIGH_DUPLICATE_PERCENTAGE",
                "severity": "medium",
                "message": f"Alto percentual de duplicatas: {duplicate_percentage:.1f}%",
                "current_value": duplicate_percentage,
                "threshold": quality_thresholds["max_duplicate_percentage"]
            })

        # Verificar colunas com baixa variabilidade
        for col in df.columns:
            unique_percentage = (df[col].nunique() / len(df)) * 100
            if unique_percentage < quality_thresholds["min_unique_percentage"]:
                validation_result["issues"].append({
                    "type": "LOW_VARIABILITY",
                    "severity": "low",
                    "message": f"Coluna '{col}' tem baixa variabilidade: {unique_percentage:.1f}% valores únicos",
                    "column_name": col,
                    "unique_percentage": unique_percentage
                })

        # Verificar outliers em colunas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
                outlier_percentage = (len(outliers) / len(series)) * 100

                outlier_analysis[col] = outlier_percentage

                if outlier_percentage > quality_thresholds["max_outlier_percentage"]:
                    validation_result["issues"].append({
                        "type": "HIGH_OUTLIER_PERCENTAGE",
                        "severity": "medium",
                        "message": f"Coluna '{col}' tem muitos outliers: {outlier_percentage:.1f}%",
                        "column_name": col,
                        "outlier_percentage": outlier_percentage
                    })

        validation_result["metrics"]["outlier_analysis"] = outlier_analysis

        # Calcular score de qualidade
        quality_score = 100.0

        # Penalizar por nulls
        quality_score -= min(null_percentage * 0.5, 25)

        # Penalizar por duplicatas
        quality_score -= min(duplicate_percentage * 0.3, 15)

        # Penalizar por outliers
        avg_outlier_percentage = np.mean(list(outlier_analysis.values())) if outlier_analysis else 0
        quality_score -= min(avg_outlier_percentage * 0.2, 10)

        validation_result["quality_score"] = max(0, quality_score)

        return validation_result

    except Exception as e:
        return {
            "valid": False,
            "error": f"Erro na validação da qualidade: {str(e)}",
            "error_code": "VALIDATION_ERROR"
        }


def validate_analysis_request(request: Dict[str, Any], df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Valida uma requisição de análise

    Args:
        request: Dict com parâmetros da requisição
        df: DataFrame opcional para validações contextuais

    Returns:
        Dict com resultado da validação
    """
    try:
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "normalized_request": {}
        }

        required_fields = ["question", "analysis_type"]

        # Verificar campos obrigatórios
        for field in required_fields:
            if field not in request:
                validation_result["errors"].append({
                    "type": "MISSING_REQUIRED_FIELD",
                    "message": f"Campo obrigatório ausente: {field}",
                    "field": field
                })
                validation_result["valid"] = False

        # Validar pergunta
        if "question" in request:
            question = request["question"]
            if not isinstance(question, str) or len(question.strip()) < 5:
                validation_result["errors"].append({
                    "type": "INVALID_QUESTION",
                    "message": "Pergunta deve ser uma string com pelo menos 5 caracteres",
                    "current_value": question
                })
                validation_result["valid"] = False

        # Validar tipo de análise
        if "analysis_type" in request:
            analysis_type = request["analysis_type"]
            valid_types = [
                "descriptive", "correlation", "outlier", "trend", 
                "comparison", "clustering", "exploratory"
            ]

            if analysis_type not in valid_types:
                validation_result["errors"].append({
                    "type": "INVALID_ANALYSIS_TYPE",
                    "message": f"Tipo de análise inválido: {analysis_type}",
                    "valid_types": valid_types,
                    "current_value": analysis_type
                })
                validation_result["valid"] = False

        # Validar colunas especificadas (se df fornecido)
        if df is not None and "columns" in request:
            specified_columns = request["columns"]
            if isinstance(specified_columns, list):
                invalid_columns = [col for col in specified_columns if col not in df.columns]
                if invalid_columns:
                    validation_result["errors"].append({
                        "type": "INVALID_COLUMNS",
                        "message": f"Colunas não existem no dataset: {invalid_columns}",
                        "invalid_columns": invalid_columns,
                        "available_columns": list(df.columns)
                    })
                    validation_result["valid"] = False

        # Validar parâmetros opcionais
        if "parameters" in request:
            params = request["parameters"]

            # Validar sample_size
            if "sample_size" in params:
                sample_size = params["sample_size"]
                if not isinstance(sample_size, int) or sample_size < 1:
                    validation_result["warnings"].append({
                        "type": "INVALID_SAMPLE_SIZE",
                        "message": f"sample_size deve ser inteiro positivo, usando padrão",
                        "current_value": sample_size
                    })

            # Validar confidence_level
            if "confidence_level" in params:
                conf_level = params["confidence_level"]
                if not isinstance(conf_level, (int, float)) or not (0 < conf_level < 1):
                    validation_result["warnings"].append({
                        "type": "INVALID_CONFIDENCE_LEVEL",
                        "message": f"confidence_level deve estar entre 0 e 1, usando padrão",
                        "current_value": conf_level
                    })

        # Criar versão normalizada da requisição
        validation_result["normalized_request"] = {
            "question": request.get("question", "").strip(),
            "analysis_type": request.get("analysis_type", "exploratory"),
            "columns": request.get("columns", []),
            "parameters": request.get("parameters", {}),
            "timestamp": datetime.now().isoformat()
        }

        return validation_result

    except Exception as e:
        return {
            "valid": False,
            "error": f"Erro na validação da requisição: {str(e)}",
            "error_code": "VALIDATION_ERROR"
        }


def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida configuração do sistema

    Args:
        config: Dict com configurações

    Returns:
        Dict com resultado da validação
    """
    try:
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_optional": []
        }

        # Configurações obrigatórias
        required_configs = [
            "gemini_api_key",
            "supabase_url",
            "supabase_anon_key"
        ]

        for config_key in required_configs:
            if config_key not in config or not config[config_key]:
                validation_result["errors"].append({
                    "type": "MISSING_REQUIRED_CONFIG",
                    "message": f"Configuração obrigatória ausente: {config_key}",
                    "config_key": config_key
                })
                validation_result["valid"] = False

        # Configurações opcionais mas recomendadas
        optional_configs = [
            "openai_api_key",
            "max_csv_size_mb",
            "max_analysis_iterations"
        ]

        for config_key in optional_configs:
            if config_key not in config:
                validation_result["missing_optional"].append(config_key)

        # Validar URLs
        url_configs = ["supabase_url"]
        for url_key in url_configs:
            if url_key in config:
                url = config[url_key]
                if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    validation_result["errors"].append({
                        "type": "INVALID_URL",
                        "message": f"URL inválida para {url_key}: {url}",
                        "config_key": url_key
                    })
                    validation_result["valid"] = False

        # Validar valores numéricos
        numeric_configs = {
            "max_csv_size_mb": {"min": 1, "max": 1000},
            "max_analysis_iterations": {"min": 1, "max": 20}
        }

        for config_key, limits in numeric_configs.items():
            if config_key in config:
                value = config[config_key]
                if not isinstance(value, (int, float)):
                    validation_result["warnings"].append({
                        "type": "INVALID_NUMERIC_CONFIG",
                        "message": f"{config_key} deve ser numérico",
                        "config_key": config_key,
                        "current_value": value
                    })
                elif not (limits["min"] <= value <= limits["max"]):
                    validation_result["warnings"].append({
                        "type": "CONFIG_OUT_OF_RANGE",
                        "message": f"{config_key} fora do intervalo recomendado: {limits['min']}-{limits['max']}",
                        "config_key": config_key,
                        "current_value": value,
                        "recommended_range": limits
                    })

        return validation_result

    except Exception as e:
        return {
            "valid": False,
            "error": f"Erro na validação da configuração: {str(e)}",
            "error_code": "VALIDATION_ERROR"
        }


def validate_memory_operation(operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida operações de memória

    Args:
        operation: Tipo de operação ("store", "search", "update", "delete")
        parameters: Parâmetros da operação

    Returns:
        Dict com resultado da validação
    """
    try:
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        valid_operations = ["store", "search", "update", "delete"]

        if operation not in valid_operations:
            validation_result["errors"].append({
                "type": "INVALID_OPERATION",
                "message": f"Operação inválida: {operation}",
                "valid_operations": valid_operations
            })
            validation_result["valid"] = False
            return validation_result

        # Validações específicas por operação
        if operation == "store":
            required_params = ["content", "analysis_type", "dataset_name"]
            for param in required_params:
                if param not in parameters:
                    validation_result["errors"].append({
                        "type": "MISSING_STORE_PARAMETER",
                        "message": f"Parâmetro obrigatório para store: {param}",
                        "parameter": param
                    })
                    validation_result["valid"] = False

        elif operation == "search":
            if "query" not in parameters:
                validation_result["errors"].append({
                    "type": "MISSING_SEARCH_QUERY",
                    "message": "Parâmetro 'query' obrigatório para search"
                })
                validation_result["valid"] = False

        elif operation == "update":
            if "memory_id" not in parameters:
                validation_result["errors"].append({
                    "type": "MISSING_MEMORY_ID",
                    "message": "Parâmetro 'memory_id' obrigatório para update"
                })
                validation_result["valid"] = False

        elif operation == "delete":
            if "memory_id" not in parameters:
                validation_result["errors"].append({
                    "type": "MISSING_MEMORY_ID",
                    "message": "Parâmetro 'memory_id' obrigatório para delete"
                })
                validation_result["valid"] = False

        return validation_result

    except Exception as e:
        return {
            "valid": False,
            "error": f"Erro na validação da operação de memória: {str(e)}",
            "error_code": "VALIDATION_ERROR"
        }


def create_validation_report(validations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cria relatório consolidado de validações

    Args:
        validations: Lista de resultados de validação

    Returns:
        Relatório consolidado
    """
    report = {
        "overall_valid": True,
        "total_validations": len(validations),
        "passed_validations": 0,
        "failed_validations": 0,
        "total_errors": 0,
        "total_warnings": 0,
        "validation_summary": [],
        "critical_issues": [],
        "recommendations": []
    }

    for i, validation in enumerate(validations):
        if validation.get("valid", False):
            report["passed_validations"] += 1
        else:
            report["failed_validations"] += 1
            report["overall_valid"] = False

        # Contar erros e warnings
        errors = validation.get("errors", [])
        warnings = validation.get("warnings", [])

        report["total_errors"] += len(errors)
        report["total_warnings"] += len(warnings)

        # Adicionar ao resumo
        validation_summary = {
            "validation_index": i,
            "valid": validation.get("valid", False),
            "error_count": len(errors),
            "warning_count": len(warnings)
        }

        if "error" in validation:
            validation_summary["error"] = validation["error"]

        report["validation_summary"].append(validation_summary)

        # Identificar issues críticos
        for error in errors:
            if error.get("severity") == "high" or error.get("type") in ["FILE_NOT_FOUND", "UNSUPPORTED_FORMAT"]:
                report["critical_issues"].append({
                    "validation_index": i,
                    "issue": error
                })

    # Gerar recomendações
    if report["failed_validations"] > 0:
        report["recommendations"].append("Corrija os erros de validação antes de prosseguir")

    if report["total_warnings"] > 0:
        report["recommendations"].append("Revise os warnings para melhorar a qualidade dos dados")

    if report["critical_issues"]:
        report["recommendations"].append("Priorize a correção dos issues críticos identificados")

    return report
