"""
Funções auxiliares e utilitários para o sistema de agentes CSV
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import re
import chardet
from pathlib import Path
import json
from datetime import datetime, timedelta
import hashlib
import base64

logger = logging.getLogger(__name__)

def format_file_size(size_bytes: int) -> str:
    """
    Formata tamanho de arquivo em formato legível

    Args:
        size_bytes: Tamanho em bytes

    Returns:
        String formatada (ex: "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"


def detect_encoding(file_path: str, sample_size: int = 10000) -> str:
    """
    Detecta automaticamente o encoding de um arquivo

    Args:
        file_path: Caminho para o arquivo
        sample_size: Tamanho da amostra para detecção

    Returns:
        String com o encoding detectado
    """
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(sample_size)
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')
    except Exception as e:
        logger.warning(f"Erro na detecção de encoding: {e}")
        return 'utf-8'


def detect_delimiter(file_path: str, encoding: str = 'utf-8', sample_lines: int = 5) -> str:
    """
    Detecta automaticamente o delimitador de um arquivo CSV

    Args:
        file_path: Caminho para o arquivo
        encoding: Encoding do arquivo
        sample_lines: Número de linhas para análise

    Returns:
        Delimitador detectado
    """
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            sample = []
            for _ in range(sample_lines):
                line = file.readline()
                if not line:
                    break
                sample.append(line)

        # Testar delimitadores comuns
        delimiters = [',', ';', '\t', '|', ':']
        delimiter_scores = {}

        for delimiter in delimiters:
            scores = []
            for line in sample:
                count = line.count(delimiter)
                scores.append(count)

            # Consistência e quantidade
            if scores:
                avg_count = np.mean(scores)
                consistency = 1 - (np.std(scores) / (avg_count + 1))
                delimiter_scores[delimiter] = avg_count * consistency

        # Retornar delimitador com melhor score
        if delimiter_scores:
            best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
            return best_delimiter

        return ','  # Default

    except Exception as e:
        logger.warning(f"Erro na detecção de delimitador: {e}")
        return ','


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa nomes de colunas removendo caracteres especiais e espaços

    Args:
        df: DataFrame para limpar

    Returns:
        DataFrame com colunas renomeadas
    """
    df_cleaned = df.copy()

    new_columns = []
    for col in df.columns:
        # Converter para string
        clean_col = str(col)

        # Remover espaços no início e fim
        clean_col = clean_col.strip()

        # Substituir espaços por underscore
        clean_col = re.sub(r'\s+', '_', clean_col)

        # Remover caracteres especiais (manter apenas letras, números, underscore)
        clean_col = re.sub(r'[^a-zA-Z0-9_]', '', clean_col)

        # Garantir que não comece com número
        if clean_col and clean_col[0].isdigit():
            clean_col = 'col_' + clean_col

        # Se ficou vazio, usar nome padrão
        if not clean_col:
            clean_col = f'column_{len(new_columns)}'

        new_columns.append(clean_col)

    df_cleaned.columns = new_columns
    return df_cleaned


def infer_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infere tipos de dados mais apropriados para cada coluna

    Args:
        df: DataFrame para análise

    Returns:
        Dict mapeando coluna -> tipo sugerido
    """
    suggestions = {}

    for col in df.columns:
        series = df[col].dropna()

        if len(series) == 0:
            suggestions[col] = 'object'
            continue

        # Verificar se é numérico
        if pd.api.types.is_numeric_dtype(series):
            # Verificar se pode ser inteiro
            if series.dtype in ['float64', 'float32']:
                if series.apply(lambda x: x.is_integer() if pd.notnull(x) else True).all():
                    suggestions[col] = 'int64'
                else:
                    suggestions[col] = 'float64'
            else:
                suggestions[col] = str(series.dtype)

        # Verificar se é data/hora
        elif series.dtype == 'object':
            # Tentar converter para datetime
            try:
                pd.to_datetime(series.head(100), infer_datetime_format=True)
                suggestions[col] = 'datetime64[ns]'
            except:
                # Verificar se é categórico
                unique_ratio = series.nunique() / len(series)
                if unique_ratio < 0.05 or series.nunique() < 20:
                    suggestions[col] = 'category'
                else:
                    suggestions[col] = 'object'

        else:
            suggestions[col] = str(series.dtype)

    return suggestions


def optimize_dataframe_memory(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Otimiza uso de memória do DataFrame

    Args:
        df: DataFrame para otimizar

    Returns:
        Tuple com DataFrame otimizado e relatório de otimização
    """
    df_optimized = df.copy()
    optimization_report = {
        'original_memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'optimizations': {},
        'memory_saved_mb': 0
    }

    for col in df.columns:
        original_dtype = df[col].dtype
        original_memory = df[col].memory_usage(deep=True)

        # Otimizar colunas numéricas
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].dtype == 'int64':
                # Tentar reduzir para int32, int16, int8
                max_val = df[col].max()
                min_val = df[col].min()

                if min_val >= -128 and max_val <= 127:
                    df_optimized[col] = df[col].astype('int8')
                elif min_val >= -32768 and max_val <= 32767:
                    df_optimized[col] = df[col].astype('int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    df_optimized[col] = df[col].astype('int32')

            elif df[col].dtype == 'float64':
                # Tentar reduzir para float32
                df_optimized[col] = pd.to_numeric(df[col], downcast='float')

        # Otimizar colunas categóricas
        elif df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Se menos de 50% valores únicos
                df_optimized[col] = df[col].astype('category')

        # Calcular economia
        new_memory = df_optimized[col].memory_usage(deep=True)
        memory_saved = original_memory - new_memory

        if memory_saved > 0:
            optimization_report['optimizations'][col] = {
                'original_dtype': str(original_dtype),
                'new_dtype': str(df_optimized[col].dtype),
                'memory_saved_bytes': memory_saved
            }

    # Calcular totais
    final_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
    optimization_report['final_memory_mb'] = final_memory
    optimization_report['memory_saved_mb'] = optimization_report['original_memory_mb'] - final_memory
    optimization_report['reduction_percentage'] = (optimization_report['memory_saved_mb'] / optimization_report['original_memory_mb']) * 100

    return df_optimized, optimization_report


def generate_data_fingerprint(df: pd.DataFrame) -> str:
    """
    Gera uma "impressão digital" única para um DataFrame

    Args:
        df: DataFrame para gerar fingerprint

    Returns:
        String hash única representando o DataFrame
    """
    # Criar string representativa do DataFrame
    fingerprint_data = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'first_row_hash': hashlib.md5(str(df.iloc[0].to_dict()).encode()).hexdigest() if len(df) > 0 else '',
        'last_row_hash': hashlib.md5(str(df.iloc[-1].to_dict()).encode()).hexdigest() if len(df) > 0 else ''
    }

    # Criar hash da representação
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    fingerprint_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()

    return fingerprint_hash[:16]  # Retornar primeiros 16 caracteres


def create_sample_dataset(dataset_type: str = "sales", n_rows: int = 1000) -> pd.DataFrame:
    """
    Cria dataset de exemplo para testes

    Args:
        dataset_type: Tipo de dataset ("sales", "customers", "products")
        n_rows: Número de linhas

    Returns:
        DataFrame de exemplo
    """
    np.random.seed(42)

    if dataset_type == "sales":
        data = {
            'data': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
            'produto': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
            'categoria': np.random.choice(['Eletrônicos', 'Roupas', 'Casa', 'Esporte'], n_rows),
            'vendas': np.random.normal(1000, 200, n_rows),
            'preco': np.random.uniform(10, 500, n_rows),
            'quantidade': np.random.poisson(5, n_rows),
            'desconto': np.random.uniform(0, 0.3, n_rows),
            'regiao': np.random.choice(['Norte', 'Sul', 'Leste', 'Oeste'], n_rows),
            'vendedor': np.random.choice([f'Vendedor_{i}' for i in range(1, 21)], n_rows)
        }

        # Adicionar correlação realista
        df = pd.DataFrame(data)
        df['receita'] = df['vendas'] * df['preco'] * (1 - df['desconto'])
        df['lucro'] = df['receita'] * np.random.uniform(0.1, 0.3, n_rows)

        # Adicionar alguns outliers
        outlier_indices = np.random.choice(df.index, size=int(n_rows * 0.02), replace=False)
        df.loc[outlier_indices, 'vendas'] *= 5

        return df

    elif dataset_type == "customers":
        data = {
            'cliente_id': range(1, n_rows + 1),
            'idade': np.random.normal(35, 12, n_rows).astype(int),
            'genero': np.random.choice(['M', 'F'], n_rows),
            'cidade': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Brasília'], n_rows),
            'renda_mensal': np.random.lognormal(8, 0.5, n_rows),
            'anos_cliente': np.random.exponential(3, n_rows),
            'score_credito': np.random.normal(650, 100, n_rows),
            'produtos_comprados': np.random.poisson(8, n_rows),
            'valor_total_compras': np.random.exponential(2000, n_rows)
        }

        df = pd.DataFrame(data)

        # Garantir idades válidas
        df['idade'] = np.clip(df['idade'], 18, 80)
        df['score_credito'] = np.clip(df['score_credito'], 300, 850)

        return df

    elif dataset_type == "products":
        categorias = ['Eletrônicos', 'Roupas', 'Casa', 'Esporte', 'Livros']
        data = {
            'produto_id': range(1, n_rows + 1),
            'nome_produto': [f'Produto_{i}' for i in range(1, n_rows + 1)],
            'categoria': np.random.choice(categorias, n_rows),
            'preco_custo': np.random.uniform(10, 200, n_rows),
            'preco_venda': np.random.uniform(20, 400, n_rows),
            'estoque': np.random.poisson(50, n_rows),
            'fornecedor': np.random.choice([f'Fornecedor_{i}' for i in range(1, 11)], n_rows),
            'peso_kg': np.random.exponential(1, n_rows),
            'avaliacoes': np.random.uniform(1, 5, n_rows),
            'vendas_mes': np.random.poisson(20, n_rows)
        }

        df = pd.DataFrame(data)

        # Garantir que preço de venda > preço de custo
        df['preco_venda'] = np.maximum(df['preco_venda'], df['preco_custo'] * 1.2)
        df['margem_lucro'] = (df['preco_venda'] - df['preco_custo']) / df['preco_venda']

        return df

    else:
        raise ValueError(f"Tipo de dataset não suportado: {dataset_type}")


def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula score de qualidade dos dados

    Args:
        df: DataFrame para avaliar

    Returns:
        Dict com scores de qualidade
    """
    scores = {}

    # Score de completude (% de valores não-nulos)
    completeness = ((df.count().sum() / (len(df) * len(df.columns))) * 100)
    scores['completeness'] = min(100, completeness)

    # Score de consistência (baseado em tipos de dados)
    type_consistency = 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Verificar se há valores infinitos ou NaN
            invalid_values = df[col].isin([np.inf, -np.inf]).sum()
            consistency = (1 - (invalid_values / len(df))) * 100
            type_consistency += consistency
        else:
            # Para colunas não-numéricas, verificar consistência de formato
            type_consistency += 100  # Simplificado

    scores['consistency'] = type_consistency / len(df.columns)

    # Score de unicidade (baseado em duplicatas)
    duplicate_ratio = df.duplicated().sum() / len(df)
    scores['uniqueness'] = (1 - duplicate_ratio) * 100

    # Score de validade (simplificado - baseado em outliers)
    validity_scores = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        validity = (1 - (len(outliers) / len(df))) * 100
        validity_scores.append(validity)

    scores['validity'] = np.mean(validity_scores) if validity_scores else 100

    # Score geral (média ponderada)
    weights = {'completeness': 0.3, 'consistency': 0.25, 'uniqueness': 0.25, 'validity': 0.2}
    overall_score = sum(scores[metric] * weight for metric, weight in weights.items())
    scores['overall'] = overall_score

    # Classificação qualitativa
    if overall_score >= 90:
        scores['quality_level'] = 'Excelente'
    elif overall_score >= 80:
        scores['quality_level'] = 'Boa'
    elif overall_score >= 70:
        scores['quality_level'] = 'Aceitável'
    elif overall_score >= 60:
        scores['quality_level'] = 'Baixa'
    else:
        scores['quality_level'] = 'Muito Baixa'

    return scores


def format_number(number: Union[int, float], precision: int = 2, 
                 use_thousands_sep: bool = True) -> str:
    """
    Formata números para exibição legível

    Args:
        number: Número para formatar
        precision: Número de casas decimais
        use_thousands_sep: Se deve usar separador de milhares

    Returns:
        String formatada
    """
    if pd.isna(number):
        return "N/A"

    if isinstance(number, (int, np.integer)):
        if use_thousands_sep:
            return f"{number:,}"
        else:
            return str(number)

    elif isinstance(number, (float, np.floating)):
        if use_thousands_sep:
            return f"{number:,.{precision}f}"
        else:
            return f"{number:.{precision}f}"

    else:
        return str(number)


def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Cria resumo estatístico abrangente

    Args:
        df: DataFrame para sumarizar

    Returns:
        Dict com estatísticas resumidas
    """
    summary = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        },
        'data_types': df.dtypes.value_counts().to_dict(),
        'missing_data': {
            'total_missing': df.isnull().sum().sum(),
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        },
        'duplicates': {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
    }

    # Estatísticas para colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()

    # Estatísticas para colunas categóricas
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        summary['categorical_summary'] = {}
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'frequency_of_most_frequent': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            }

    return summary


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divisão segura que trata divisão por zero

    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor padrão se denominador for zero

    Returns:
        Resultado da divisão ou valor padrão
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except:
        return default


def encode_categorical_simple(df: pd.DataFrame, columns: List[str] = None, 
                            method: str = 'label') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Codifica variáveis categóricas de forma simples

    Args:
        df: DataFrame com variáveis categóricas
        columns: Lista de colunas para codificar (None = todas categóricas)
        method: Método de codificação ('label' ou 'onehot')

    Returns:
        Tuple com DataFrame codificado e informações da codificação
    """
    df_encoded = df.copy()
    encoding_info = {}

    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'label':
            # Label encoding
            unique_values = df[col].unique()
            label_map = {val: idx for idx, val in enumerate(unique_values)}
            df_encoded[col] = df[col].map(label_map)

            encoding_info[col] = {
                'method': 'label',
                'mapping': label_map,
                'original_unique_count': len(unique_values)
            }

        elif method == 'onehot':
            # One-hot encoding (simplificado)
            dummies = pd.get_dummies(df[col], prefix=col)
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)

            encoding_info[col] = {
                'method': 'onehot',
                'new_columns': dummies.columns.tolist(),
                'original_unique_count': len(df[col].unique())
            }

    return df_encoded, encoding_info


def generate_timestamp() -> str:
    """
    Gera timestamp formatado para logs e identificação

    Returns:
        String com timestamp
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def mask_sensitive_data(text: str, patterns: List[str] = None) -> str:
    """
    Mascara dados sensíveis em texto

    Args:
        text: Texto para mascarar
        patterns: Lista de padrões regex para mascarar

    Returns:
        Texto com dados mascarados
    """
    if patterns is None:
        patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]

    masked_text = text
    for pattern in patterns:
        masked_text = re.sub(pattern, '***MASKED***', masked_text)

    return masked_text
