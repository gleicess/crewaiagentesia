"""
DataLoader Agent - Especialista em ingestão e validação de dados CSV
"""
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from ..config.prompts import DATA_LOADER_PROMPT
from ..config.settings import settings
from ..tools.csv_tools import (
    load_and_validate_csv,
    analyze_column_types, 
    generate_quality_report,
    sample_data_for_analysis,
    convert_data_types
)
from ..utils.validators import validate_file_format, validate_file_size
from ..utils.helpers import format_file_size, detect_encoding

logger = logging.getLogger(__name__)

class DataLoaderAgent:
    """Agente especializado em carregamento e validação de dados CSV"""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.1  # Baixa temperatura para precisão
        )

        self.agent = Agent(
            role=DATA_LOADER_PROMPT["role"],
            goal=DATA_LOADER_PROMPT["goal"],
            backstory=DATA_LOADER_PROMPT["backstory"],
            instructions=DATA_LOADER_PROMPT["instructions"],
            llm=self.llm,
            tools=[
                load_and_validate_csv,
                analyze_column_types,
                generate_quality_report,
                sample_data_for_analysis,
                convert_data_types
            ],
            verbose=True,
            memory=True,
            max_iter=3
        )

    def load_dataset(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Carrega e valida um dataset CSV completo

        Args:
            file_path: Caminho para o arquivo
            **kwargs: Parâmetros adicionais (encoding, delimiter, etc.)

        Returns:
            Dict com dados carregados e relatório de validação
        """
        try:
            # Validar arquivo antes de carregar
            file_validation = self._validate_file(file_path)
            if not file_validation["valid"]:
                return {
                    "success": False,
                    "error": file_validation["error"],
                    "file_path": file_path
                }

            # Carregar dados usando ferramenta especializada
            load_result = load_and_validate_csv(
                file_path=file_path,
                encoding=kwargs.get("encoding"),
                delimiter=kwargs.get("delimiter")
            )

            if not load_result.get("success"):
                return load_result

            # Extrair DataFrame dos dados carregados
            df = pd.DataFrame(load_result["data"])

            # Análise detalhada dos tipos de colunas
            column_analysis = analyze_column_types(df)

            # Relatório de qualidade dos dados
            quality_report = generate_quality_report(df)

            # Amostra para análise (se dataset muito grande)
            sample_df = sample_data_for_analysis(df, sample_size=kwargs.get("sample_size", 1000))

            # Gerar insights sobre a estrutura dos dados
            structure_insights = self._generate_structure_insights(df, column_analysis, quality_report)

            # Sugestões de pré-processamento
            preprocessing_suggestions = self._suggest_preprocessing(quality_report, column_analysis)

            return {
                "success": True,
                "file_path": file_path,
                "dataset_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict(),
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "encoding": load_result.get("encoding"),
                    "delimiter": load_result.get("delimiter")
                },
                "column_analysis": column_analysis,
                "quality_report": quality_report,
                "structure_insights": structure_insights,
                "preprocessing_suggestions": preprocessing_suggestions,
                "sample_data": sample_df.to_dict('records')[:100],  # Primeiros 100 para preview
                "data": df,  # DataFrame completo para outros agentes
                "timestamp": pd.Timestamp.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro no DataLoader Agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def _validate_file(self, file_path: str) -> Dict[str, Any]:
        """Valida arquivo antes do carregamento"""
        try:
            # Verificar se arquivo existe
            if not Path(file_path).exists():
                return {"valid": False, "error": f"Arquivo não encontrado: {file_path}"}

            # Validar formato
            format_validation = validate_file_format(file_path)
            if not format_validation["valid"]:
                return format_validation

            # Validar tamanho
            size_validation = validate_file_size(file_path, max_size_mb=settings.max_csv_size_mb)
            if not size_validation["valid"]:
                return size_validation

            return {"valid": True}

        except Exception as e:
            return {"valid": False, "error": f"Erro na validação: {str(e)}"}

    def _generate_structure_insights(self, df: pd.DataFrame, column_analysis: Dict, quality_report: Dict) -> List[str]:
        """Gera insights sobre a estrutura dos dados"""
        insights = []

        # Insights sobre tamanho
        total_rows, total_cols = df.shape
        insights.append(f"Dataset contém {total_rows:,} registros e {total_cols} colunas")

        # Insights sobre tipos de dados
        numeric_cols = sum(1 for col_info in column_analysis.values() if col_info["semantic_type"] == "numerical")
        categorical_cols = sum(1 for col_info in column_analysis.values() if col_info["semantic_type"] == "categorical")
        datetime_cols = sum(1 for col_info in column_analysis.values() if col_info["semantic_type"] == "datetime")
        text_cols = sum(1 for col_info in column_analysis.values() if col_info["semantic_type"] == "text")

        if numeric_cols > 0:
            insights.append(f"{numeric_cols} colunas numéricas identificadas - adequadas para análises estatísticas")

        if categorical_cols > 0:
            insights.append(f"{categorical_cols} colunas categóricas encontradas - úteis para segmentação e agrupamentos")

        if datetime_cols > 0:
            insights.append(f"{datetime_cols} colunas de data/hora detectadas - permitem análises temporais")

        if text_cols > 0:
            insights.append(f"{text_cols} colunas de texto identificadas - podem requerer processamento especial")

        # Insights sobre qualidade
        null_percentage = quality_report.get("null_percentage", 0)
        if null_percentage > 20:
            insights.append(f"Alto percentual de valores ausentes ({null_percentage:.1f}%) - considere estratégias de imputação")
        elif null_percentage > 5:
            insights.append(f"Percentual moderado de valores ausentes ({null_percentage:.1f}%)")
        else:
            insights.append("Baixo percentual de valores ausentes - boa qualidade dos dados")

        # Insights sobre duplicatas
        duplicate_percentage = quality_report.get("duplicate_percentage", 0)
        if duplicate_percentage > 10:
            insights.append(f"Alto número de duplicatas ({duplicate_percentage:.1f}%) - recomenda-se remoção")
        elif duplicate_percentage > 1:
            insights.append(f"Algumas duplicatas detectadas ({duplicate_percentage:.1f}%)")

        # Insights sobre outliers
        outliers_analysis = quality_report.get("outliers_analysis", {})
        if outliers_analysis:
            high_outlier_cols = [col for col, info in outliers_analysis.items() 
                               if info.get("outlier_percentage", 0) > 10]
            if high_outlier_cols:
                insights.append(f"Colunas com muitos outliers detectados: {', '.join(high_outlier_cols[:3])}")

        return insights

    def _suggest_preprocessing(self, quality_report: Dict, column_analysis: Dict) -> List[str]:
        """Sugere etapas de pré-processamento"""
        suggestions = []

        # Sugestões baseadas na qualidade
        if quality_report.get("duplicate_percentage", 0) > 1:
            suggestions.append("Remover registros duplicados para evitar viés nas análises")

        if quality_report.get("completely_null_rows", 0) > 0:
            suggestions.append("Remover linhas completamente vazias")

        if quality_report.get("completely_null_columns", 0) > 0:
            suggestions.append("Remover colunas completamente vazias")

        # Sugestões baseadas nos tipos de dados
        for col, col_info in column_analysis.items():
            null_pct = col_info.get("null_percentage", 0)

            if null_pct > 50:
                suggestions.append(f"Considerar remoção da coluna '{col}' (>50% valores ausentes)")
            elif null_pct > 20:
                if col_info["semantic_type"] == "numerical":
                    suggestions.append(f"Imputar valores ausentes na coluna '{col}' (mediana ou média)")
                elif col_info["semantic_type"] == "categorical":
                    suggestions.append(f"Imputar valores ausentes na coluna '{col}' (moda ou categoria 'Outros')")

        # Sugestões sobre conversões de tipos
        for col, col_info in column_analysis.items():
            if col_info["semantic_type"] == "datetime" and col_info["pandas_dtype"] == "object":
                suggestions.append(f"Converter coluna '{col}' para tipo datetime para análises temporais")

            if col_info["semantic_type"] == "categorical" and col_info["unique_percentage"] < 5:
                suggestions.append(f"Converter coluna '{col}' para tipo category para otimização de memória")

        # Sugestões sobre outliers
        outliers_analysis = quality_report.get("outliers_analysis", {})
        for col, outlier_info in outliers_analysis.items():
            outlier_pct = outlier_info.get("outlier_percentage", 0)
            if outlier_pct > 15:
                suggestions.append(f"Investigar outliers na coluna '{col}' ({outlier_pct:.1f}% dos dados)")

        return suggestions


def create_data_loader_agent() -> Agent:
    """Factory function para criar o agente DataLoader"""
    data_loader = DataLoaderAgent()
    return data_loader.agent
