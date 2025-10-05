"""
Visualization Agent - Especialista em criação de visualizações interativas com Plotly
"""
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import logging
import json
from datetime import datetime

from ..config.prompts import VISUALIZATION_PROMPT
from ..config.settings import settings
from ..tools.plotting_tools import (
    create_histogram,
    create_scatter_plot,
    create_box_plot,
    create_correlation_heatmap,
    create_time_series,
    create_bar_chart
)

logger = logging.getLogger(__name__)

# Configurar tema padrão
pio.templates.default = "plotly_white"

class VisualizationAgent:
    """Agente especializado em criação de visualizações interativas"""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.2  # Baixa temperatura para consistência
        )

        self.agent = Agent(
            role=VISUALIZATION_PROMPT["role"],
            goal=VISUALIZATION_PROMPT["goal"],
            backstory=VISUALIZATION_PROMPT["backstory"],
            instructions=VISUALIZATION_PROMPT["instructions"],
            llm=self.llm,
            tools=[
                create_histogram,
                create_scatter_plot,
                create_box_plot,
                create_correlation_heatmap,
                create_time_series,
                create_bar_chart
            ],
            verbose=True,
            memory=True,
            max_iter=3
        )

        # Paleta de cores personalizada
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

    def create_comprehensive_dashboard(self, df: pd.DataFrame, analysis_results: Dict[str, Any], 
                                     visualization_request: str = "overview") -> Dict[str, Any]:
        """
        Cria dashboard completo de visualizações baseado nos dados e análises

        Args:
            df: DataFrame com os dados
            analysis_results: Resultados da análise do Analyzer Agent
            visualization_request: Tipo de visualização solicitada

        Returns:
            Dict com todas as visualizações criadas
        """
        try:
            dashboard = {
                "request": visualization_request,
                "timestamp": datetime.now().isoformat(),
                "visualizations": [],
                "summary": {},
                "interactive_features": [],
                "insights": []
            }

            # Determinar tipos de visualização necessários
            viz_types = self._determine_visualization_types(df, analysis_results, visualization_request)

            # Criar visualizações baseadas no tipo de requisição
            for viz_type in viz_types:
                viz_result = self._create_visualization_by_type(df, analysis_results, viz_type)
                if viz_result and "figure" in viz_result:
                    dashboard["visualizations"].append(viz_result)

            # Criar visualizações específicas baseadas nos resultados da análise
            if "correlation_analysis" in analysis_results:
                corr_viz = self._create_correlation_visualizations(df, analysis_results["correlation_analysis"])
                if corr_viz:
                    dashboard["visualizations"].extend(corr_viz)

            if "outlier_detection" in analysis_results:
                outlier_viz = self._create_outlier_visualizations(df, analysis_results["outlier_detection"])
                if outlier_viz:
                    dashboard["visualizations"].extend(outlier_viz)

            if "trend_analysis" in analysis_results:
                trend_viz = self._create_trend_visualizations(df, analysis_results["trend_analysis"])
                if trend_viz:
                    dashboard["visualizations"].extend(trend_viz)

            if "clustering_analysis" in analysis_results:
                cluster_viz = self._create_clustering_visualizations(df, analysis_results["clustering_analysis"])
                if cluster_viz:
                    dashboard["visualizations"].extend(cluster_viz)

            # Gerar insights visuais
            dashboard["insights"] = self._generate_visual_insights(dashboard["visualizations"])

            # Criar resumo do dashboard
            dashboard["summary"] = self._create_dashboard_summary(dashboard["visualizations"])

            return dashboard

        except Exception as e:
            logger.error(f"Erro na criação do dashboard: {e}")
            return {
                "error": str(e),
                "request": visualization_request,
                "timestamp": datetime.now().isoformat()
            }

    def _determine_visualization_types(self, df: pd.DataFrame, analysis_results: Dict, 
                                     request: str) -> List[str]:
        """Determina quais tipos de visualização criar"""
        viz_types = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        if request == "overview" or request == "general":
            # Dashboard overview completo
            if numeric_cols:
                viz_types.extend(["distributions", "correlations"])
            if categorical_cols:
                viz_types.append("categorical_analysis")
            if datetime_cols and numeric_cols:
                viz_types.append("time_series")
            viz_types.append("outlier_detection")

        elif request == "correlation":
            if len(numeric_cols) >= 2:
                viz_types.extend(["correlation_matrix", "scatter_plots"])

        elif request == "distribution":
            if numeric_cols:
                viz_types.extend(["histograms", "box_plots"])

        elif request == "outliers":
            if numeric_cols:
                viz_types.extend(["box_plots", "outlier_scatter"])

        elif request == "trends":
            if datetime_cols and numeric_cols:
                viz_types.append("time_series")

        elif request == "comparison":
            if categorical_cols and numeric_cols:
                viz_types.append("categorical_comparison")

        return viz_types

    def _create_visualization_by_type(self, df: pd.DataFrame, analysis_results: Dict, 
                                    viz_type: str) -> Optional[Dict[str, Any]]:
        """Cria visualização específica por tipo"""
        try:
            if viz_type == "distributions":
                return self._create_distribution_overview(df)

            elif viz_type == "correlations":
                return self._create_correlation_overview(df)

            elif viz_type == "categorical_analysis":
                return self._create_categorical_overview(df)

            elif viz_type == "outlier_detection":
                return self._create_outlier_overview(df)

            elif viz_type == "time_series":
                return self._create_time_series_overview(df)

            return None

        except Exception as e:
            logger.error(f"Erro na criação de visualização {viz_type}: {e}")
            return None

    def _create_distribution_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria overview das distribuições"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return None

        # Criar subplots para múltiplas distribuições
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) - 1) // n_cols + 1

        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.08
        )

        for i, col in enumerate(numeric_cols):
            row = (i // n_cols) + 1
            col_pos = (i % n_cols) + 1

            # Criar histograma
            data = df[col].dropna()
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=col,
                    showlegend=False,
                    marker_color=self.colors[i % len(self.colors)]
                ),
                row=row, col=col_pos
            )

        fig.update_layout(
            title="Distribuições das Variáveis Numéricas",
            height=300 * n_rows,
            template="plotly_white"
        )

        return {
            "type": "distribution_overview",
            "title": "Overview das Distribuições",
            "figure": fig,
            "columns_analyzed": numeric_cols,
            "insights": [f"Análise de distribuição para {len(numeric_cols)} variáveis numéricas"]
        }

    def _create_correlation_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria overview das correlações"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return None

        # Matriz de correlação
        corr_matrix = numeric_df.corr()

        # Criar heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Correlação")
        ))

        fig.update_layout(
            title="Matriz de Correlação",
            xaxis_title="",
            yaxis_title="",
            width=600,
            height=600,
            template="plotly_white"
        )

        # Encontrar correlações mais fortes
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": corr_value
                    })

        return {
            "type": "correlation_overview",
            "title": "Matriz de Correlação",
            "figure": fig,
            "strong_correlations": strong_correlations,
            "insights": [f"Correlação entre {len(corr_matrix.columns)} variáveis numéricas"]
        }

    def _create_categorical_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria overview das variáveis categóricas"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            return None

        # Selecionar até 4 colunas categóricas para visualizar
        cols_to_plot = categorical_cols[:4]

        n_cols = min(2, len(cols_to_plot))
        n_rows = (len(cols_to_plot) - 1) // n_cols + 1

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=cols_to_plot,
            specs=[[{"type": "domain"}] * n_cols for _ in range(n_rows)]
        )

        for i, col in enumerate(cols_to_plot):
            row = (i // n_cols) + 1
            col_pos = (i % n_cols) + 1

            # Contar valores
            value_counts = df[col].value_counts().head(10)

            # Criar gráfico de pizza
            fig.add_trace(
                go.Pie(
                    labels=value_counts.index,
                    values=value_counts.values,
                    name=col,
                    showlegend=False
                ),
                row=row, col=col_pos
            )

        fig.update_layout(
            title="Distribuição das Variáveis Categóricas",
            height=400 * n_rows,
            template="plotly_white"
        )

        return {
            "type": "categorical_overview",
            "title": "Overview das Categorias",
            "figure": fig,
            "columns_analyzed": cols_to_plot,
            "insights": [f"Distribuição categórica para {len(cols_to_plot)} variáveis"]
        }

    def _create_outlier_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria overview dos outliers"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return None

        # Selecionar até 6 colunas para boxplots
        cols_to_plot = numeric_cols[:6]

        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) - 1) // n_cols + 1

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=cols_to_plot
        )

        for i, col in enumerate(cols_to_plot):
            row = (i // n_cols) + 1
            col_pos = (i % n_cols) + 1

            fig.add_trace(
                go.Box(
                    y=df[col].dropna(),
                    name=col,
                    showlegend=False,
                    marker_color=self.colors[i % len(self.colors)]
                ),
                row=row, col=col_pos
            )

        fig.update_layout(
            title="Detecção de Outliers (Box Plots)",
            height=300 * n_rows,
            template="plotly_white"
        )

        return {
            "type": "outlier_overview",
            "title": "Overview dos Outliers",
            "figure": fig,
            "columns_analyzed": cols_to_plot,
            "insights": [f"Detecção de outliers para {len(cols_to_plot)} variáveis"]
        }

    def _create_time_series_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria overview das séries temporais"""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not datetime_cols or not numeric_cols:
            return None

        # Usar primeira coluna de data e até 3 colunas numéricas
        date_col = datetime_cols[0]
        value_cols = numeric_cols[:3]

        fig = go.Figure()

        for i, col in enumerate(value_cols):
            # Preparar dados
            temp_df = df[[date_col, col]].dropna().sort_values(date_col)

            fig.add_trace(go.Scatter(
                x=temp_df[date_col],
                y=temp_df[col],
                mode='lines+markers',
                name=col,
                line=dict(color=self.colors[i % len(self.colors)])
            ))

        fig.update_layout(
            title=f"Séries Temporais - {', '.join(value_cols)}",
            xaxis_title=date_col,
            yaxis_title="Valor",
            template="plotly_white",
            hovermode='x unified'
        )

        return {
            "type": "time_series_overview",
            "title": "Overview das Séries Temporais",
            "figure": fig,
            "date_column": date_col,
            "value_columns": value_cols,
            "insights": [f"Análise temporal de {len(value_cols)} variáveis"]
        }

    def _create_correlation_visualizations(self, df: pd.DataFrame, 
                                         correlation_results: Dict) -> List[Dict[str, Any]]:
        """Cria visualizações específicas para correlações"""
        visualizations = []

        # Scatter plots para correlações fortes
        strong_corrs = correlation_results.get("strong_correlations", [])

        for i, corr in enumerate(strong_corrs[:3]):  # Top 3 correlações
            var1, var2 = corr["variable_1"], corr["variable_2"]

            fig = px.scatter(
                df, x=var1, y=var2,
                title=f"Correlação: {var1} vs {var2} (r={corr['correlation']:.3f})",
                template="plotly_white",
                trendline="ols"
            )

            fig.update_traces(marker=dict(color=self.colors[i % len(self.colors)]))

            visualizations.append({
                "type": "correlation_scatter",
                "title": f"Scatter Plot: {var1} vs {var2}",
                "figure": fig,
                "correlation": corr["correlation"],
                "insights": [f"Correlação {corr['strength']} {corr['direction']} entre as variáveis"]
            })

        return visualizations

    def _create_outlier_visualizations(self, df: pd.DataFrame, 
                                     outlier_results: Dict) -> List[Dict[str, Any]]:
        """Cria visualizações específicas para outliers"""
        visualizations = []

        # Encontrar colunas com mais outliers
        high_outlier_cols = []
        for col, outlier_info in outlier_results.items():
            if isinstance(outlier_info, dict) and "iqr_method" in outlier_info:
                if outlier_info["iqr_method"]["outlier_percentage"] > 5:
                    high_outlier_cols.append((col, outlier_info))

        # Criar scatter plots destacando outliers
        for col, outlier_info in high_outlier_cols[:2]:  # Top 2
            fig = go.Figure()

            # Pontos normais
            normal_data = df[col].copy()
            outlier_values = outlier_info["iqr_method"]["outlier_values"]

            # Criar máscara para outliers
            is_outlier = normal_data.isin(outlier_values)

            # Pontos normais
            fig.add_trace(go.Scatter(
                x=normal_data[~is_outlier].index,
                y=normal_data[~is_outlier],
                mode='markers',
                name='Valores Normais',
                marker=dict(color='blue', size=6)
            ))

            # Outliers
            fig.add_trace(go.Scatter(
                x=normal_data[is_outlier].index,
                y=normal_data[is_outlier],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=8, symbol='x')
            ))

            fig.update_layout(
                title=f"Outliers em {col} ({outlier_info['iqr_method']['outlier_percentage']:.1f}%)",
                xaxis_title="Índice",
                yaxis_title=col,
                template="plotly_white"
            )

            visualizations.append({
                "type": "outlier_scatter",
                "title": f"Outliers: {col}",
                "figure": fig,
                "outlier_count": outlier_info["iqr_method"]["outlier_count"],
                "outlier_percentage": outlier_info["iqr_method"]["outlier_percentage"],
                "insights": [f"{outlier_info['iqr_method']['outlier_count']} outliers detectados"]
            })

        return visualizations

    def _create_trend_visualizations(self, df: pd.DataFrame, 
                                   trend_results: Dict) -> List[Dict[str, Any]]:
        """Cria visualizações específicas para tendências"""
        visualizations = []

        for trend_name, trend_info in trend_results.items():
            if isinstance(trend_info, dict) and "trend_direction" in trend_info:
                # Extrair nomes das colunas
                parts = trend_name.split("_vs_")
                if len(parts) == 2:
                    date_col, value_col = parts

                    # Preparar dados
                    temp_df = df[[date_col, value_col]].dropna().sort_values(date_col)

                    fig = go.Figure()

                    # Linha principal
                    fig.add_trace(go.Scatter(
                        x=temp_df[date_col],
                        y=temp_df[value_col],
                        mode='lines+markers',
                        name=value_col,
                        line=dict(color='blue')
                    ))

                    # Linha de tendência
                    if len(temp_df) > 2:
                        z = np.polyfit(range(len(temp_df)), temp_df[value_col], 1)
                        trend_line = np.poly1d(z)(range(len(temp_df)))

                        fig.add_trace(go.Scatter(
                            x=temp_df[date_col],
                            y=trend_line,
                            mode='lines',
                            name='Tendência',
                            line=dict(color='red', dash='dash')
                        ))

                    fig.update_layout(
                        title=f"Tendência: {value_col} ao longo do tempo",
                        xaxis_title=date_col,
                        yaxis_title=value_col,
                        template="plotly_white"
                    )

                    visualizations.append({
                        "type": "trend_analysis",
                        "title": f"Tendência: {value_col}",
                        "figure": fig,
                        "trend_direction": trend_info["trend_direction"],
                        "trend_strength": trend_info["trend_strength"],
                        "insights": [f"Tendência {trend_info['trend_direction']} com força {trend_info['trend_strength']}"]
                    })

        return visualizations

    def _create_clustering_visualizations(self, df: pd.DataFrame, 
                                        clustering_results: Dict) -> List[Dict[str, Any]]:
        """Cria visualizações para clustering"""
        visualizations = []

        if "cluster_assignments" not in clustering_results:
            return visualizations

        # Usar PCA para visualização 2D se há mais de 2 dimensões
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        cluster_labels = clustering_results["cluster_assignments"]

        if numeric_df.shape[1] >= 2:
            try:
                from sklearn.decomposition import PCA

                if numeric_df.shape[1] > 2:
                    # Reduzir para 2D com PCA
                    pca = PCA(n_components=2)
                    coords_2d = pca.fit_transform(numeric_df)
                    x_coords, y_coords = coords_2d[:, 0], coords_2d[:, 1]
                    x_label, y_label = "Componente Principal 1", "Componente Principal 2"
                else:
                    # Usar as duas primeiras colunas numéricas
                    x_coords = numeric_df.iloc[:, 0]
                    y_coords = numeric_df.iloc[:, 1]
                    x_label, y_label = numeric_df.columns[0], numeric_df.columns[1]

                # Criar scatter plot dos clusters
                fig = go.Figure()

                unique_clusters = set(cluster_labels)
                for cluster_id in unique_clusters:
                    mask = np.array(cluster_labels) == cluster_id

                    fig.add_trace(go.Scatter(
                        x=x_coords[mask],
                        y=y_coords[mask],
                        mode='markers',
                        name=f'Cluster {cluster_id}',
                        marker=dict(
                            color=self.colors[cluster_id % len(self.colors)],
                            size=8
                        )
                    ))

                fig.update_layout(
                    title=f"Clusters Identificados ({clustering_results['optimal_clusters']} grupos)",
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    template="plotly_white"
                )

                visualizations.append({
                    "type": "clustering",
                    "title": "Análise de Clusters",
                    "figure": fig,
                    "n_clusters": clustering_results["optimal_clusters"],
                    "insights": [f"{clustering_results['optimal_clusters']} clusters distintos identificados"]
                })

            except Exception as e:
                logger.warning(f"Erro na visualização de clustering: {e}")

        return visualizations

    def _generate_visual_insights(self, visualizations: List[Dict]) -> List[str]:
        """Gera insights baseados nas visualizações criadas"""
        insights = []

        # Contar tipos de visualização
        viz_types = [viz.get("type", "unknown") for viz in visualizations]

        insights.append(f"Dashboard com {len(visualizations)} visualizações interativas criadas")

        # Insights específicos por tipo
        correlation_viz = [v for v in visualizations if v.get("type") == "correlation_scatter"]
        if correlation_viz:
            strong_corrs = [v for v in correlation_viz if abs(v.get("correlation", 0)) > 0.8]
            if strong_corrs:
                insights.append(f"{len(strong_corrs)} correlações muito fortes visualizadas")

        outlier_viz = [v for v in visualizations if v.get("type") == "outlier_scatter"]
        if outlier_viz:
            total_outliers = sum(v.get("outlier_count", 0) for v in outlier_viz)
            insights.append(f"{total_outliers} outliers identificados e visualizados")

        trend_viz = [v for v in visualizations if v.get("type") == "trend_analysis"]
        if trend_viz:
            increasing = [v for v in trend_viz if v.get("trend_direction") == "increasing"]
            decreasing = [v for v in trend_viz if v.get("trend_direction") == "decreasing"]
            if increasing:
                insights.append(f"{len(increasing)} tendências crescentes visualizadas")
            if decreasing:
                insights.append(f"{len(decreasing)} tendências decrescentes visualizadas")

        cluster_viz = [v for v in visualizations if v.get("type") == "clustering"]
        if cluster_viz:
            n_clusters = cluster_viz[0].get("n_clusters", 0)
            insights.append(f"Dados organizados em {n_clusters} grupos distintos")

        return insights

    def _create_dashboard_summary(self, visualizations: List[Dict]) -> Dict[str, Any]:
        """Cria resumo do dashboard"""
        return {
            "total_visualizations": len(visualizations),
            "visualization_types": list(set(viz.get("type", "unknown") for viz in visualizations)),
            "interactive_features": [
                "Hover para detalhes",
                "Zoom e pan",
                "Legendas clicáveis",
                "Export PNG/HTML"
            ],
            "recommended_actions": [
                "Explore as visualizações interativamente",
                "Use hover para ver valores específicos",
                "Clique nas legendas para filtrar dados",
                "Faça zoom em áreas de interesse"
            ]
        }


def create_visualization_agent() -> Agent:
    """Factory function para criar o agente Visualization"""
    visualization = VisualizationAgent()
    return visualization.agent
