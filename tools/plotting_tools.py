"""
Ferramentas para criação de visualizações interativas com Plotly
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import json
from pathlib import Path

from crewai_tools import tool

logger = logging.getLogger(__name__)

# Configurar tema padrão
pio.templates.default = "plotly_white"

# Paleta de cores personalizada
CUSTOM_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


@tool("create_histogram")
def create_histogram(
    df: pd.DataFrame, 
    column: str, 
    title: Optional[str] = None,
    bins: int = 30,
    color: Optional[str] = None,
    template: str = "plotly_white"
) -> Dict[str, Any]:
    """
    Cria histograma interativo para uma coluna numérica.

    Args:
        df: DataFrame com os dados
        column: Nome da coluna para o histograma
        title: Título do gráfico
        bins: Número de bins
        color: Cor das barras
        template: Template do Plotly

    Returns:
        Dict com o gráfico e metadados
    """
    if column not in df.columns:
        return {"error": f"Coluna '{column}' não encontrada"}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Coluna '{column}' não é numérica"}

    # Remover valores nulos
    data = df[column].dropna()

    if len(data) == 0:
        return {"error": f"Coluna '{column}' não tem dados válidos"}

    # Criar histograma
    fig = px.histogram(
        data, 
        x=column,
        nbins=bins,
        title=title or f"Distribuição de {column}",
        template=template,
        color_discrete_sequence=[color] if color else CUSTOM_COLORS
    )

    # Personalizar layout
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequência",
        showlegend=False,
        hovermode='x unified'
    )

    # Adicionar estatísticas ao hover
    fig.update_traces(
        hovertemplate=f"<b>{column}</b>: %{{x}}<br>" +
                     "Frequência: %{y}<br>" +
                     "<extra></extra>"
    )

    # Calcular estatísticas
    stats = {
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "min": data.min(),
        "max": data.max(),
        "count": len(data)
    }

    return {
        "figure": fig,
        "type": "histogram",
        "column": column,
        "statistics": stats,
        "insights": generate_histogram_insights(data, stats)
    }


@tool("create_scatter_plot")
def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    title: Optional[str] = None,
    template: str = "plotly_white"
) -> Dict[str, Any]:
    """
    Cria scatter plot interativo.

    Args:
        df: DataFrame com os dados
        x_column: Coluna para eixo X
        y_column: Coluna para eixo Y
        color_column: Coluna para colorir pontos
        size_column: Coluna para tamanho dos pontos
        title: Título do gráfico
        template: Template do Plotly

    Returns:
        Dict com o gráfico e metadados
    """
    # Verificar colunas
    required_cols = [x_column, y_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {"error": f"Colunas não encontradas: {missing_cols}"}

    # Verificar se são numéricas
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"Coluna '{col}' deve ser numérica"}

    # Remover linhas com valores nulos nas colunas principais
    plot_data = df[[x_column, y_column]].dropna()
    if color_column and color_column in df.columns:
        plot_data = df[[x_column, y_column, color_column]].dropna()
    if size_column and size_column in df.columns:
        plot_data = df[[x_column, y_column, color_column, size_column]].dropna() if color_column else df[[x_column, y_column, size_column]].dropna()

    if len(plot_data) == 0:
        return {"error": "Não há dados válidos para o gráfico"}

    # Criar scatter plot
    fig = px.scatter(
        plot_data,
        x=x_column,
        y=y_column,
        color=color_column,
        size=size_column,
        title=title or f"{y_column} vs {x_column}",
        template=template,
        color_discrete_sequence=CUSTOM_COLORS
    )

    # Personalizar layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        hovermode='closest'
    )

    # Calcular correlação se ambas são numéricas
    correlation = plot_data[x_column].corr(plot_data[y_column]) if len(plot_data) > 1 else None

    return {
        "figure": fig,
        "type": "scatter",
        "x_column": x_column,
        "y_column": y_column,
        "correlation": correlation,
        "data_points": len(plot_data),
        "insights": generate_scatter_insights(plot_data, x_column, y_column, correlation)
    }


@tool("create_box_plot")
def create_box_plot(
    df: pd.DataFrame,
    y_column: str,
    x_column: Optional[str] = None,
    title: Optional[str] = None,
    template: str = "plotly_white"
) -> Dict[str, Any]:
    """
    Cria box plot para detectar outliers e distribuições.

    Args:
        df: DataFrame com os dados
        y_column: Coluna numérica para análise
        x_column: Coluna categórica para agrupamento (opcional)
        title: Título do gráfico
        template: Template do Plotly

    Returns:
        Dict com o gráfico e metadados
    """
    if y_column not in df.columns:
        return {"error": f"Coluna '{y_column}' não encontrada"}

    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return {"error": f"Coluna '{y_column}' deve ser numérica"}

    # Preparar dados
    plot_data = df[[y_column]].copy()
    if x_column and x_column in df.columns:
        plot_data[x_column] = df[x_column]
        plot_data = plot_data.dropna()
    else:
        plot_data = plot_data.dropna()

    if len(plot_data) == 0:
        return {"error": "Não há dados válidos para o gráfico"}

    # Criar box plot
    fig = px.box(
        plot_data,
        x=x_column,
        y=y_column,
        title=title or f"Box Plot de {y_column}",
        template=template,
        color=x_column if x_column else None,
        color_discrete_sequence=CUSTOM_COLORS
    )

    # Personalizar layout
    fig.update_layout(
        xaxis_title=x_column or "",
        yaxis_title=y_column,
        showlegend=bool(x_column)
    )

    # Detectar outliers
    Q1 = plot_data[y_column].quantile(0.25)
    Q3 = plot_data[y_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = plot_data[(plot_data[y_column] < lower_bound) | (plot_data[y_column] > upper_bound)]

    return {
        "figure": fig,
        "type": "box",
        "y_column": y_column,
        "x_column": x_column,
        "outlier_count": len(outliers),
        "outlier_percentage": (len(outliers) / len(plot_data)) * 100,
        "quartiles": {"Q1": Q1, "Q3": Q3, "IQR": IQR},
        "insights": generate_boxplot_insights(plot_data, y_column, len(outliers))
    }


@tool("create_correlation_heatmap")
def create_correlation_heatmap(
    df: pd.DataFrame,
    title: Optional[str] = None,
    template: str = "plotly_white"
) -> Dict[str, Any]:
    """
    Cria heatmap de correlações entre variáveis numéricas.

    Args:
        df: DataFrame com os dados
        title: Título do gráfico
        template: Template do Plotly

    Returns:
        Dict com o gráfico e metadados
    """
    # Selecionar apenas colunas numéricas
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return {"error": "É necessário pelo menos 2 colunas numéricas"}

    # Calcular correlações
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
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>" +
                     "Correlação: %{z:.3f}<br>" +
                     "<extra></extra>"
    ))

    # Personalizar layout
    fig.update_layout(
        title=title or "Matriz de Correlação",
        template=template,
        xaxis_title="",
        yaxis_title="",
        width=600,
        height=600
    )

    # Encontrar correlações mais fortes
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if not np.isnan(corr_value):
                corr_pairs.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": corr_value
                })

    # Ordenar por correlação absoluta
    corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "figure": fig,
        "type": "correlation_heatmap",
        "correlation_matrix": corr_matrix.to_dict(),
        "strongest_correlations": corr_pairs[:10],
        "insights": generate_correlation_insights(corr_pairs)
    }


@tool("create_time_series")
def create_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_columns: List[str],
    title: Optional[str] = None,
    template: str = "plotly_white"
) -> Dict[str, Any]:
    """
    Cria gráfico de séries temporais.

    Args:
        df: DataFrame com os dados
        date_column: Coluna com datas
        value_columns: Lista de colunas numéricas para plotar
        title: Título do gráfico
        template: Template do Plotly

    Returns:
        Dict com o gráfico e metadados
    """
    # Verificar colunas
    all_columns = [date_column] + value_columns
    missing_cols = [col for col in all_columns if col not in df.columns]
    if missing_cols:
        return {"error": f"Colunas não encontradas: {missing_cols}"}

    # Preparar dados
    plot_data = df[all_columns].copy()

    # Converter coluna de data
    try:
        plot_data[date_column] = pd.to_datetime(plot_data[date_column])
    except:
        return {"error": f"Não foi possível converter '{date_column}' para data"}

    # Remover valores nulos
    plot_data = plot_data.dropna()

    if len(plot_data) == 0:
        return {"error": "Não há dados válidos para o gráfico"}

    # Ordenar por data
    plot_data = plot_data.sort_values(date_column)

    # Criar gráfico
    fig = go.Figure()

    for i, col in enumerate(value_columns):
        fig.add_trace(go.Scatter(
            x=plot_data[date_column],
            y=plot_data[col],
            mode='lines+markers',
            name=col,
            line=dict(color=CUSTOM_COLORS[i % len(CUSTOM_COLORS)]),
            hovertemplate=f"<b>{col}</b><br>" +
                         "Data: %{x}<br>" +
                         "Valor: %{y}<br>" +
                         "<extra></extra>"
        ))

    # Personalizar layout
    fig.update_layout(
        title=title or f"Série Temporal - {', '.join(value_columns)}",
        xaxis_title=date_column,
        yaxis_title="Valor",
        template=template,
        hovermode='x unified'
    )

    return {
        "figure": fig,
        "type": "time_series",
        "date_column": date_column,
        "value_columns": value_columns,
        "data_points": len(plot_data),
        "date_range": {
            "start": plot_data[date_column].min().isoformat(),
            "end": plot_data[date_column].max().isoformat()
        },
        "insights": generate_timeseries_insights(plot_data, date_column, value_columns)
    }


@tool("create_bar_chart")  
def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    orientation: str = "v",
    title: Optional[str] = None,
    template: str = "plotly_white"
) -> Dict[str, Any]:
    """
    Cria gráfico de barras.

    Args:
        df: DataFrame com os dados
        x_column: Coluna para eixo X (categórica)
        y_column: Coluna para eixo Y (numérica)
        orientation: Orientação ('v' vertical, 'h' horizontal)
        title: Título do gráfico
        template: Template do Plotly

    Returns:
        Dict com o gráfico e metadados
    """
    if x_column not in df.columns or y_column not in df.columns:
        return {"error": f"Colunas não encontradas"}

    # Agrupar dados se necessário
    if df[x_column].dtype == 'object' or df[x_column].dtype.name == 'category':
        plot_data = df.groupby(x_column)[y_column].sum().reset_index()
    else:
        plot_data = df[[x_column, y_column]].dropna()

    # Criar gráfico de barras
    if orientation == "h":
        fig = px.bar(
            plot_data,
            x=y_column,
            y=x_column,
            orientation="h",
            title=title or f"{y_column} por {x_column}",
            template=template,
            color_discrete_sequence=CUSTOM_COLORS
        )
    else:
        fig = px.bar(
            plot_data,
            x=x_column,
            y=y_column,
            title=title or f"{y_column} por {x_column}",
            template=template,
            color_discrete_sequence=CUSTOM_COLORS
        )

    # Personalizar layout
    fig.update_layout(
        xaxis_title=y_column if orientation == "h" else x_column,
        yaxis_title=x_column if orientation == "h" else y_column,
        showlegend=False
    )

    return {
        "figure": fig,
        "type": "bar_chart",
        "x_column": x_column,
        "y_column": y_column,
        "categories": len(plot_data),
        "insights": generate_bar_chart_insights(plot_data, x_column, y_column)
    }


# Funções auxiliares para gerar insights
def generate_histogram_insights(data: pd.Series, stats: Dict) -> List[str]:
    """Gera insights para histogramas"""
    insights = []

    # Distribuição
    skewness = data.skew()
    if abs(skewness) < 0.5:
        insights.append("Distribuição aproximadamente simétrica")
    elif skewness > 0.5:
        insights.append("Distribuição assimétrica à direita (cauda longa à direita)")
    else:
        insights.append("Distribuição assimétrica à esquerda (cauda longa à esquerda)")

    # Amplitude
    amplitude = stats["max"] - stats["min"]
    cv = stats["std"] / stats["mean"] if stats["mean"] != 0 else 0

    if cv < 0.1:
        insights.append("Baixa variabilidade nos dados")
    elif cv > 0.3:
        insights.append("Alta variabilidade nos dados")
    else:
        insights.append("Variabilidade moderada nos dados")

    return insights


def generate_scatter_insights(data: pd.DataFrame, x_col: str, y_col: str, correlation: float) -> List[str]:
    """Gera insights para scatter plots"""
    insights = []

    if correlation is not None:
        if abs(correlation) > 0.8:
            insights.append(f"Correlação muito forte ({correlation:.3f}) entre {x_col} e {y_col}")
        elif abs(correlation) > 0.6:
            insights.append(f"Correlação forte ({correlation:.3f}) entre {x_col} e {y_col}")
        elif abs(correlation) > 0.3:
            insights.append(f"Correlação moderada ({correlation:.3f}) entre {x_col} e {y_col}")
        else:
            insights.append(f"Correlação fraca ({correlation:.3f}) entre {x_col} e {y_col}")

        if correlation > 0:
            insights.append("Relação positiva: quando uma variável aumenta, a outra tende a aumentar")
        elif correlation < 0:
            insights.append("Relação negativa: quando uma variável aumenta, a outra tende a diminuir")

    return insights


def generate_boxplot_insights(data: pd.DataFrame, y_col: str, outlier_count: int) -> List[str]:
    """Gera insights para box plots"""
    insights = []

    if outlier_count > 0:
        outlier_pct = (outlier_count / len(data)) * 100
        insights.append(f"Detectados {outlier_count} outliers ({outlier_pct:.1f}% dos dados)")

        if outlier_pct > 10:
            insights.append("Alto número de outliers - considere investigar esses valores")
        elif outlier_pct > 5:
            insights.append("Número moderado de outliers detectados")
    else:
        insights.append("Nenhum outlier detectado usando critério IQR")

    return insights


def generate_correlation_insights(corr_pairs: List[Dict]) -> List[str]:
    """Gera insights para matriz de correlação"""
    insights = []

    if corr_pairs:
        strongest = corr_pairs[0]
        insights.append(f"Correlação mais forte: {strongest['var1']} e {strongest['var2']} ({strongest['correlation']:.3f})")

        # Contar correlações por intensidade
        strong_corr = sum(1 for pair in corr_pairs if abs(pair['correlation']) > 0.7)
        if strong_corr > 0:
            insights.append(f"{strong_corr} pares de variáveis com correlação forte (>0.7)")

    return insights


def generate_timeseries_insights(data: pd.DataFrame, date_col: str, value_cols: List[str]) -> List[str]:
    """Gera insights para séries temporais"""
    insights = []

    date_range = data[date_col].max() - data[date_col].min()
    insights.append(f"Período analisado: {date_range.days} dias")

    # Verificar tendências
    for col in value_cols:
        correlation_with_time = data[date_col].astype('int64').corr(data[col])
        if correlation_with_time > 0.5:
            insights.append(f"{col} apresenta tendência crescente ao longo do tempo")
        elif correlation_with_time < -0.5:
            insights.append(f"{col} apresenta tendência decrescente ao longo do tempo")
        else:
            insights.append(f"{col} não apresenta tendência clara ao longo do tempo")

    return insights


def generate_bar_chart_insights(data: pd.DataFrame, x_col: str, y_col: str) -> List[str]:
    """Gera insights para gráficos de barras"""
    insights = []

    # Categoria com maior valor
    max_idx = data[y_col].idxmax()
    max_category = data.loc[max_idx, x_col]
    max_value = data.loc[max_idx, y_col]

    insights.append(f"Maior valor: {max_category} ({max_value})")

    # Distribuição dos valores
    cv = data[y_col].std() / data[y_col].mean() if data[y_col].mean() != 0 else 0
    if cv > 1:
        insights.append("Grande variação entre as categorias")
    elif cv < 0.3:
        insights.append("Valores relativamente uniformes entre categorias")

    return insights
