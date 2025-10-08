"""
Analyzer Agent - Especialista em análise exploratória e estatística de dados
"""
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings

from ..config.prompts import ANALYZER_PROMPT
from ..config.settings import settings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class AnalyzerAgent:
    """Agente especializado em análises estatísticas e descoberta de insights"""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.2  # Baixa temperatura para análises precisas
        )

        self.agent = Agent(
            role=ANALYZER_PROMPT["role"],
            goal=ANALYZER_PROMPT["goal"],
            backstory=ANALYZER_PROMPT["backstory"],
            instructions=ANALYZER_PROMPT["instructions"],
            llm=self.llm,
            verbose=True,
            memory=True,
            max_iter=5
        )

    def perform_comprehensive_analysis(self, df: pd.DataFrame, analysis_focus: str = "general") -> Dict[str, Any]:
        """
        Realiza análise exploratória completa dos dados

        Args:
            df: DataFrame para análise
            analysis_focus: Foco da análise ("general", "correlation", "outliers", "trends", etc.)

        Returns:
            Dict com resultados da análise completa
        """
        try:
            results = {
                "analysis_focus": analysis_focus,
                "dataset_overview": self._get_dataset_overview(df),
                "descriptive_statistics": self._calculate_descriptive_stats(df),
                "correlation_analysis": self._analyze_correlations(df),
                "outlier_detection": self._detect_outliers(df),
                "distribution_analysis": self._analyze_distributions(df),
                "insights": [],
                "recommendations": [],
                "statistical_tests": {},
                "patterns": {}
            }

            # Análises específicas baseadas no foco
            if analysis_focus == "correlation" or analysis_focus == "general":
                results["correlation_analysis"] = self._deep_correlation_analysis(
                    df)

            if analysis_focus == "outliers" or analysis_focus == "general":
                results["outlier_detection"] = self._advanced_outlier_detection(
                    df)

            if analysis_focus == "trends" or analysis_focus == "general":
                results["trend_analysis"] = self._analyze_trends(df)

            if analysis_focus == "clustering" or analysis_focus == "general":
                results["clustering_analysis"] = self._perform_clustering(df)

            # Gerar insights interpretativos
            results["insights"] = self._generate_insights(results, df)

            # Gerar recomendações
            results["recommendations"] = self._generate_recommendations(
                results, df)

            # Testes estatísticos automáticos
            results["statistical_tests"] = self._perform_statistical_tests(df)

            return results

        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_focus": analysis_focus
            }

    def _get_dataset_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Visão geral básica do dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(
            include=['datetime64']).columns.tolist()

        return {
            "shape": df.shape,
            "total_cells": df.shape[0] * df.shape[1],
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "column_types": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "datetime": len(datetime_cols)
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                "columns_with_missing": df.columns[df.isnull().any()].tolist()
            }
        }

    def _calculate_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estatísticas descritivas avançadas"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {"message": "Nenhuma coluna numérica encontrada"}

        stats_dict = {}

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) > 0:
                stats_dict[col] = {
                    "count": int(series.count()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "mode": float(series.mode().iloc[0]) if not series.mode().empty else None,
                    "std": float(series.std()),
                    "variance": float(series.var()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "range": float(series.max() - series.min()),
                    "q1": float(series.quantile(0.25)),
                    "q3": float(series.quantile(0.75)),
                    "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                    "coefficient_variation": float(series.std() / series.mean()) if series.mean() != 0 else None
                }

        return stats_dict

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise básica de correlações"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return {"message": "Menos de 2 colunas numéricas para análise de correlação"}

        # Matriz de correlação
        corr_matrix = numeric_df.corr()

        # Encontrar correlações mais fortes
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    correlations.append({
                        "variable_1": corr_matrix.columns[i],
                        "variable_2": corr_matrix.columns[j],
                        "correlation": float(corr_value),
                        "strength": self._classify_correlation_strength(abs(corr_value)),
                        "direction": "positive" if corr_value > 0 else "negative"
                    })

        # Ordenar por força da correlação
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "strong_correlations": [c for c in correlations if abs(c["correlation"]) > 0.7],
            "moderate_correlations": [c for c in correlations if 0.3 <= abs(c["correlation"]) <= 0.7],
            "weak_correlations": [c for c in correlations if abs(c["correlation"]) < 0.3],
            "all_correlations": correlations[:20]  # Top 20
        }

    def _deep_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise aprofundada de correlações"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return {"message": "Insuficientes colunas numéricas para análise profunda"}

        results = self._analyze_correlations(df)

        # Análise de correlação parcial
        try:
            from scipy.stats import pearsonr, spearmanr

            partial_correlations = {}
            columns = numeric_df.columns.tolist()

            for i, col1 in enumerate(columns):
                for j, col2 in enumerate(columns[i+1:], i+1):
                    # Correlação de Pearson
                    pearson_corr, pearson_p = pearsonr(numeric_df[col1].dropna(),
                                                       numeric_df[col2].dropna())

                    # Correlação de Spearman
                    spearman_corr, spearman_p = spearmanr(numeric_df[col1].dropna(),
                                                          numeric_df[col2].dropna())

                    partial_correlations[f"{col1}_vs_{col2}"] = {
                        "pearson": {"correlation": float(pearson_corr), "p_value": float(pearson_p)},
                        "spearman": {"correlation": float(spearman_corr), "p_value": float(spearman_p)}
                    }

            results["correlation_tests"] = partial_correlations

        except ImportError:
            logger.warning(
                "Scipy não disponível para testes de correlação avançados")

        return results

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detecção básica de outliers"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {"message": "Nenhuma coluna numérica para detecção de outliers"}

        outliers_results = {}

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) > 0:
                # Método IQR
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                iqr_outliers = series[(series < lower_bound)
                                      | (series > upper_bound)]

                # Método Z-Score
                z_scores = np.abs(stats.zscore(series))
                z_outliers = series[z_scores > 3]

                outliers_results[col] = {
                    "iqr_method": {
                        "outlier_count": len(iqr_outliers),
                        "outlier_percentage": (len(iqr_outliers) / len(series)) * 100,
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "outlier_values": iqr_outliers.tolist()[:10]  # Top 10
                    },
                    "zscore_method": {
                        "outlier_count": len(z_outliers),
                        "outlier_percentage": (len(z_outliers) / len(series)) * 100,
                        "outlier_values": z_outliers.tolist()[:10]  # Top 10
                    }
                }

        return outliers_results

    def _advanced_outlier_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detecção avançada de outliers usando Isolation Forest"""
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)

        if numeric_df.empty or len(numeric_df) < 10:
            return {"message": "Dados insuficientes para detecção avançada de outliers"}

        basic_results = self._detect_outliers(df)

        try:
            # Isolation Forest para detecção multivariada
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_predictions = iso_forest.fit_predict(numeric_df)

            # Identificar índices dos outliers
            outlier_indices = np.where(outlier_predictions == -1)[0]
            outlier_scores = iso_forest.decision_function(numeric_df)

            advanced_results = {
                "isolation_forest": {
                    "total_outliers": len(outlier_indices),
                    "outlier_percentage": (len(outlier_indices) / len(numeric_df)) * 100,
                    "outlier_indices": outlier_indices.tolist()[:20],
                    "outlier_scores": outlier_scores[outlier_indices].tolist()[:20]
                }
            }

            # Combinar com resultados básicos
            basic_results.update(advanced_results)

        except Exception as e:
            logger.warning(f"Erro na detecção avançada de outliers: {e}")

        return basic_results

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa distribuições das variáveis numéricas"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {"message": "Nenhuma coluna numérica para análise de distribuição"}

        distributions = {}

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) > 10:
                # Teste de normalidade
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(
                        series.sample(min(5000, len(series))))
                    is_normal = shapiro_p > 0.05
                except:
                    is_normal = None
                    shapiro_stat, shapiro_p = None, None

                distributions[col] = {
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                    "normality_test": {
                        "is_normal": is_normal,
                        "shapiro_statistic": float(shapiro_stat) if shapiro_stat else None,
                        "shapiro_p_value": float(shapiro_p) if shapiro_p else None
                    },
                    "distribution_shape": self._classify_distribution_shape(series.skew(), series.kurtosis())
                }

        return distributions

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa tendências temporais se houver colunas de data"""
        datetime_cols = df.select_dtypes(
            include=['datetime64']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not datetime_cols or not numeric_cols:
            return {"message": "Colunas de data e numéricas necessárias para análise de tendências"}

        trends = {}

        for date_col in datetime_cols:
            for num_col in numeric_cols:
                try:
                    # Preparar dados para análise temporal
                    temp_df = df[[date_col, num_col]
                                 ].dropna().sort_values(date_col)

                    if len(temp_df) > 10:
                        # Calcular tendência usando correlação com tempo
                        temp_df['time_numeric'] = pd.to_numeric(
                            temp_df[date_col])
                        correlation = temp_df[num_col].corr(
                            temp_df['time_numeric'])

                        # Análise de sazonalidade básica
                        temp_df['month'] = temp_df[date_col].dt.month
                        monthly_stats = temp_df.groupby(
                            'month')[num_col].agg(['mean', 'std']).round(2)

                        trends[f"{date_col}_vs_{num_col}"] = {
                            "trend_correlation": float(correlation),
                            "trend_direction": "increasing" if correlation > 0.1 else "decreasing" if correlation < -0.1 else "stable",
                            "trend_strength": self._classify_correlation_strength(abs(correlation)),
                            "monthly_patterns": monthly_stats.to_dict(),
                            "data_points": len(temp_df),
                            "time_span_days": (temp_df[date_col].max() - temp_df[date_col].min()).days
                        }

                except Exception as e:
                    logger.warning(
                        f"Erro na análise de tendência {date_col} vs {num_col}: {e}")

        return trends

    def _perform_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Realiza análise de clustering nos dados numéricos"""
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)

        if numeric_df.shape[1] < 2 or len(numeric_df) < 10:
            return {"message": "Dados insuficientes para análise de clustering"}

        try:
            # Padronizar dados
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            # Determinar número ótimo de clusters (método elbow)
            inertias = []
            k_range = range(2, min(11, len(numeric_df)))

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)

            # Escolher k ótimo (simplificado)
            optimal_k = k_range[0] if len(inertias) > 0 else 3

            # Clustering final
            final_kmeans = KMeans(n_clusters=optimal_k,
                                  random_state=42, n_init=10)
            cluster_labels = final_kmeans.fit_predict(scaled_data)

            # Análise dos clusters
            cluster_stats = {}
            for cluster_id in range(optimal_k):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = numeric_df[cluster_mask]

                cluster_stats[f"cluster_{cluster_id}"] = {
                    "size": int(cluster_mask.sum()),
                    "percentage": float((cluster_mask.sum() / len(numeric_df)) * 100),
                    "centroid": cluster_data.mean().round(2).to_dict(),
                    "characteristics": self._describe_cluster(cluster_data, numeric_df)
                }

            return {
                "optimal_clusters": optimal_k,
                "cluster_assignments": cluster_labels.tolist(),
                "cluster_statistics": cluster_stats,
                "silhouette_score": self._calculate_silhouette_score(scaled_data, cluster_labels),
                "features_used": numeric_df.columns.tolist()
            }

        except Exception as e:
            logger.error(f"Erro no clustering: {e}")
            return {"error": str(e)}

    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Realiza testes estatísticos automáticos"""
        tests = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Testes de normalidade para cada coluna numérica
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 10:
                try:
                    # Teste de Shapiro-Wilk
                    shapiro_stat, shapiro_p = stats.shapiro(
                        series.sample(min(5000, len(series))))

                    tests[f"normality_{col}"] = {
                        "test": "Shapiro-Wilk",
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": shapiro_p > 0.05,
                        "interpretation": "Normal" if shapiro_p > 0.05 else "Não normal"
                    }
                except Exception as e:
                    logger.warning(
                        f"Erro no teste de normalidade para {col}: {e}")

        return tests

    def _generate_insights(self, results: Dict, df: pd.DataFrame) -> List[str]:
        """Gera insights interpretativos baseados nos resultados"""
        insights = []

        # Insights sobre correlações
        if "correlation_analysis" in results and "strong_correlations" in results["correlation_analysis"]:
            strong_corrs = results["correlation_analysis"]["strong_correlations"]
            if strong_corrs:
                top_corr = strong_corrs[0]
                insights.append(
                    f"Correlação mais forte encontrada entre {top_corr['variable_1']} e {top_corr['variable_2']} "
                    f"({top_corr['correlation']:.3f}) - {top_corr['direction']}"
                )

        # Insights sobre outliers
        if "outlier_detection" in results:
            high_outlier_cols = []
            for col, outlier_info in results["outlier_detection"].items():
                if isinstance(outlier_info, dict) and "iqr_method" in outlier_info:
                    if outlier_info["iqr_method"]["outlier_percentage"] > 10:
                        high_outlier_cols.append(col)

            if high_outlier_cols:
                insights.append(
                    f"Colunas com alto percentual de outliers: {', '.join(high_outlier_cols[:3])}")

        # Insights sobre distribuições
        if "distribution_analysis" in results:
            non_normal_cols = []
            for col, dist_info in results["distribution_analysis"].items():
                if isinstance(dist_info, dict) and "normality_test" in dist_info:
                    if not dist_info["normality_test"].get("is_normal", True):
                        non_normal_cols.append(col)

            if non_normal_cols:
                insights.append(
                    f"Variáveis com distribuição não-normal: {', '.join(non_normal_cols[:3])}")

        # Insights sobre clustering
        if "clustering_analysis" in results and "optimal_clusters" in results["clustering_analysis"]:
            n_clusters = results["clustering_analysis"]["optimal_clusters"]
            insights.append(
                f"Dados podem ser organizados em {n_clusters} grupos distintos")

        # Insights sobre tendências
        if "trend_analysis" in results:
            increasing_trends = []
            decreasing_trends = []
            for trend_name, trend_info in results["trend_analysis"].items():
                if isinstance(trend_info, dict):
                    direction = trend_info.get("trend_direction")
                    if direction == "increasing":
                        increasing_trends.append(trend_name)
                    elif direction == "decreasing":
                        decreasing_trends.append(trend_name)

            if increasing_trends:
                insights.append(
                    f"Tendências crescentes identificadas em: {len(increasing_trends)} relações temporais")
            if decreasing_trends:
                insights.append(
                    f"Tendências decrescentes identificadas em: {len(decreasing_trends)} relações temporais")

        return insights

    def _generate_recommendations(self, results: Dict, df: pd.DataFrame) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []

        # Recomendações sobre correlações
        if "correlation_analysis" in results:
            strong_corrs = results["correlation_analysis"].get(
                "strong_correlations", [])
            if len(strong_corrs) > 3:
                recommendations.append(
                    "Explorar as correlações fortes para identificar relações causais")
            elif len(strong_corrs) == 0:
                recommendations.append(
                    "Investigar transformações de variáveis para encontrar relações não-lineares")

        # Recomendações sobre outliers
        if "outlier_detection" in results:
            for col, outlier_info in results["outlier_detection"].items():
                if isinstance(outlier_info, dict) and "iqr_method" in outlier_info:
                    if outlier_info["iqr_method"]["outlier_percentage"] > 20:
                        recommendations.append(
                            f"Investigar e possivelmente tratar outliers na coluna {col}")

        # Recomendações sobre normalidade
        if "distribution_analysis" in results:
            for col, dist_info in results["distribution_analysis"].items():
                if isinstance(dist_info, dict) and "normality_test" in dist_info:
                    if not dist_info["normality_test"].get("is_normal", True):
                        recommendations.append(
                            f"Considerar transformações para normalizar a variável {col}")

        # Recomendações gerais
        if df.isnull().sum().sum() > 0:
            recommendations.append(
                "Implementar estratégia de tratamento para valores ausentes")

        return recommendations

    # Métodos auxiliares
    def _classify_correlation_strength(self, corr_abs: float) -> str:
        """Classifica a força da correlação"""
        if corr_abs >= 0.8:
            return "muito forte"
        elif corr_abs >= 0.6:
            return "forte"
        elif corr_abs >= 0.4:
            return "moderada"
        elif corr_abs >= 0.2:
            return "fraca"
        else:
            return "muito fraca"

    def _classify_distribution_shape(self, skewness: float, kurtosis: float) -> str:
        """Classifica a forma da distribuição"""
        if abs(skewness) < 0.5:
            skew_desc = "simétrica"
        elif skewness > 0:
            skew_desc = "assimétrica à direita"
        else:
            skew_desc = "assimétrica à esquerda"

        if abs(kurtosis) < 0.5:
            kurt_desc = "normal"
        elif kurtosis > 0:
            kurt_desc = "leptocúrtica"
        else:
            kurt_desc = "platicúrtica"

        return f"{skew_desc}, {kurt_desc}"

    def _describe_cluster(self, cluster_data: pd.DataFrame, full_data: pd.DataFrame) -> Dict[str, str]:
        """Descreve características do cluster"""
        characteristics = {}

        for col in cluster_data.columns:
            cluster_mean = cluster_data[col].mean()
            full_mean = full_data[col].mean()

            if cluster_mean > full_mean * 1.2:
                characteristics[col] = "muito acima da média"
            elif cluster_mean > full_mean * 1.1:
                characteristics[col] = "acima da média"
            elif cluster_mean < full_mean * 0.8:
                characteristics[col] = "muito abaixo da média"
            elif cluster_mean < full_mean * 0.9:
                characteristics[col] = "abaixo da média"
            else:
                characteristics[col] = "próximo da média"

        return characteristics

    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calcula silhouette score para clustering"""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(data, labels))
        except:
            return 0.0
# --- Nova função para detectar gráficos na pergunta ---


def detect_plot_intent(query: str, df: pd.DataFrame):
    plot_words = [
        ("histograma", "histogram"),
        ("dispersão", "scatter"),
        ("gráfico", "histogram"),  # default para gráfico geral
        ("correlação", "scatter")  # correlação geralmente é gráfico de dispersão
    ]
    for word, plot_type in plot_words:
        if word in query.lower():
            # Tentativa automática: pega a primeira coluna numérica encontrada
            numeric_cols = df.select_dtypes(
                include=[float, int]).columns.tolist()
            if numeric_cols:
                if plot_type == "histogram":
                    return {
                        "plot": True,
                        "plot_type": plot_type,
                        # usa a primeira como exemplo
                        "column": numeric_cols[0],
                        "x": numeric_cols[0],
                        "y": None
                    }
                elif plot_type == "scatter" and len(numeric_cols) >= 2:
                    return {
                        "plot": True,
                        "plot_type": plot_type,
                        "x": numeric_cols[0],
                        "y": numeric_cols[1],
                        "column": None
                    }
            return {"plot": False}
    return {"plot": False}


def create_analyzer_agent() -> Agent:
    """Factory function para criar o agente Analyzer"""
    analyzer = AnalyzerAgent()
    return analyzer.agent
