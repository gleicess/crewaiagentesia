"""
Aplicação Streamlit para o Sistema de Agentes CSV 
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json
import logging
from typing import Dict, List, Any, Optional
import uuid
import time

from agents.analyzer import AnalyzerAgent, detect_plot_intent


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="CSV Agent System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    .agent-status {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid;
    }

    .agent-active {
        background-color: #e8f5e8;
        border-left-color: #28a745;
    }

    .agent-completed {
        background-color: #e3f2fd;
        border-left-color: #007bff;
    }

    .agent-waiting {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }

    .insight-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Inicializa o estado da sessão"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}

    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None

    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {
            'orchestrator': 'waiting',
            'data_loader': 'waiting',
            'analyzer': 'waiting',
            'visualization': 'waiting',
            'memory': 'waiting'
        }

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}


def render_header():
    """Renderiza o cabeçalho da aplicação"""
    st.markdown('<h1 class="main-header">📊 CSV Agent System</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            🤖 Sistema de Agentes de IA para Análise Conversacional de CSV
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renderiza a barra lateral com controles"""
    with st.sidebar:
        st.header("🎛️ Controles")

        # Upload de arquivo
        st.subheader("📁 Upload do Dataset")
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type=['csv', 'xlsx', 'xls'],
            help="Suporte para CSV, Excel (xlsx, xls)"
        )

        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.uploaded_files:
                # Processar novo arquivo
                with st.spinner("Processando arquivo..."):
                    df = load_file(uploaded_file)
                    if df is not None:
                        st.session_state.uploaded_files[uploaded_file.name] = {
                            'data': df,
                            'uploaded_at': datetime.now(),
                            'size': len(df),
                            'columns': list(df.columns)
                        }
                        st.session_state.current_dataset = uploaded_file.name
                        st.success(
                            f"✅ {uploaded_file.name} carregado com sucesso!")
                        st.rerun()

        # Seleção de dataset
        if st.session_state.uploaded_files:
            st.subheader("📊 Dataset Atual")
            dataset_options = list(st.session_state.uploaded_files.keys())
            selected_dataset = st.selectbox(
                "Selecione o dataset para análise:",
                dataset_options,
                index=0 if st.session_state.current_dataset is None else dataset_options.index(
                    st.session_state.current_dataset)
            )
            st.session_state.current_dataset = selected_dataset

            # Informações do dataset
            dataset_info = st.session_state.uploaded_files[selected_dataset]
            st.info(f"""
            📏 **Dimensões:** {dataset_info['size']} linhas x {len(dataset_info['columns'])} colunas  
            📅 **Carregado:** {dataset_info['uploaded_at'].strftime('%d/%m/%Y %H:%M')}
            """)

        # Status dos agentes
        st.subheader("🤖 Status dos Agentes")
        render_agent_status()

        # Configurações
        st.subheader("⚙️ Configurações")

        st.slider("Máx. iterações por agente", 1, 10, 5, key="max_iterations")
        st.slider("Limite de outliers (%)", 1, 20, 10, key="outlier_threshold")

        analysis_depth = st.selectbox(
            "Profundidade da análise:",
            ["Rápida", "Padrão", "Profunda"],
            index=1
        )
        st.session_state.analysis_depth = analysis_depth


def render_agent_status():
    """Renderiza o status dos agentes"""
    agent_names = {
        'orchestrator': '🎯 Orchestrator',
        'data_loader': '📊 DataLoader',
        'analyzer': '🔍 Analyzer',
        'visualization': '📈 Visualization',
        'memory': '🧠 Memory'
    }

    for agent_key, agent_name in agent_names.items():
        status = st.session_state.agent_status[agent_key]

        if status == 'active':
            st.markdown(f"""
            <div class="agent-status agent-active">
                🟢 {agent_name} - Ativo
            </div>
            """, unsafe_allow_html=True)
        elif status == 'completed':
            st.markdown(f"""
            <div class="agent-status agent-completed">
                ✅ {agent_name} - Concluído
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="agent-status agent-waiting">
                ⏳ {agent_name} - Aguardando
            </div>
            """, unsafe_allow_html=True)


def load_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Carrega arquivo uploaded"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Detectar encoding
            bytes_data = uploaded_file.getvalue()

            # Tentar diferentes encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(io.BytesIO(bytes_data), encoding=encoding)
                    break
                except:
                    continue

            if df is None:
                st.error("Não foi possível determinar o encoding do arquivo CSV")
                return None

        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado")
            return None

        return df

    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None


def render_main_interface():
    """Renderiza a interface principal"""

    if not st.session_state.uploaded_files:
        # Página de boas-vindas
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown("""
            ### 🚀 Bem-vindo ao CSV Agent System!

            Para começar:
            1. **📁 Faça upload** de um arquivo CSV ou Excel na barra lateral
            2. **❓ Faça uma pergunta** sobre seus dados
            3. **🤖 Deixe os agentes** trabalharem para você!

            #### 💡 Exemplos de perguntas:
            - "Qual a distribuição da coluna idade?"
            - "Existe correlação entre preço e qualidade?" 
            - "Me mostre os outliers nos dados"
            - "Compare as vendas por categoria"
            - "Analise as tendências temporais"
            """)

            # Demonstração com dados de exemplo
            if st.button("🎲 Testar com dados de exemplo", use_container_width=True):
                sample_data = create_sample_data()
                st.session_state.uploaded_files["exemplo_vendas.csv"] = {
                    'data': sample_data,
                    'uploaded_at': datetime.now(),
                    'size': len(sample_data),
                    'columns': list(sample_data.columns)
                }
                st.session_state.current_dataset = "exemplo_vendas.csv"
                st.rerun()

    else:
        # Interface principal com dados carregados
        render_analysis_interface()


def create_sample_data() -> pd.DataFrame:
    """Cria dados de exemplo para demonstração"""
    import numpy as np

    np.random.seed(42)
    n_samples = 1000

    data = {
        'data': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'categoria': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'vendas': np.random.normal(1000, 200, n_samples),
        'preco': np.random.uniform(10, 100, n_samples),
        'qualidade': np.random.uniform(1, 10, n_samples),
        'regiao': np.random.choice(['Norte', 'Sul', 'Leste', 'Oeste'], n_samples)
    }

    df = pd.DataFrame(data)

    # Adicionar correlação entre preço e qualidade
    df['qualidade'] = df['preco'] * 0.08 + np.random.normal(0, 1, n_samples)
    df['qualidade'] = np.clip(df['qualidade'], 1, 10)

    # Adicionar alguns outliers
    outlier_indices = np.random.choice(df.index, 50, replace=False)
    df.loc[outlier_indices, 'vendas'] *= 3

    return df


def render_analysis_interface():
    """Renderiza interface de análise"""

    current_data = st.session_state.uploaded_files[st.session_state.current_dataset]['data']

    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Análise Conversacional",
        "📊 Preview dos Dados",
        "📈 Visualizações",
        "🧠 Memória de Análises"
    ])

    with tab1:
        render_conversational_analysis(current_data)

    with tab2:
        render_data_preview(current_data)

    with tab3:
        render_visualizations()

    with tab4:
        render_analysis_memory()


def render_conversational_analysis(df: pd.DataFrame):
    """Renderiza interface de análise conversacional"""

    st.subheader("💬 Faça uma pergunta sobre seus dados")

    # Campo de pergunta
    user_question = st.text_input(
        "Sua pergunta:",
        placeholder="Ex: Qual a correlação entre preço e qualidade?",
        key="user_question"
    )

    # Sugestões de perguntas
    with st.expander("💡 Sugestões de perguntas"):
        col1, col2 = st.columns(2)

        suggestions = [
            "Descreva estatisticamente os dados",
            "Existe correlação entre as variáveis numéricas?",
            "Quais são os outliers nos dados?",
            "Compare as categorias por vendas",
            "Analise as tendências ao longo do tempo",
            "Qual a distribuição da variável target?",
            "Existem padrões sazonais?",
            "Identifique valores anômalos"
        ]

        with col1:
            for suggestion in suggestions[:4]:
                if st.button(f"📝 {suggestion}", key=f"sugg_{suggestion}"):
                    st.session_state.user_question = suggestion
                    st.rerun()

        with col2:
            for suggestion in suggestions[4:]:
                if st.button(f"📝 {suggestion}", key=f"sugg_{suggestion}"):
                    st.session_state.user_question = suggestion
                    st.rerun()

    # Botão de análise
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        analyze_button = st.button(
            "🚀 Analisar",
            disabled=not user_question,
            use_container_width=True,
            type="primary"
        )

    if analyze_button and user_question:
    plot_intent = detect_plot_intent(user_question, df)
    if plot_intent["plot"]:
        import plotly.express as px
        if plot_intent["plot_type"] == "histogram":
            fig = px.histogram(df, x=plot_intent["column"][0])
            st.plotly_chart(fig)
        elif plot_intent["plot_type"] == "scatter":
            x_col = plot_intent["x"][0]
            y_col = plot_intent["y"][1] if len(
                plot_intent["y"]) > 1 else plot_intent["y"][0]
            fig = px.scatter(df, x=x_col, y=y_col)
            st.plotly_chart(fig)
    else:
        perform_analysis(user_question, df)

    # Mostrar resultados se existirem
    if st.session_state.analysis_results:
        render_analysis_results()


def perform_analysis(question: str, df: pd.DataFrame):
    """Executa a análise usando os agentes"""

    # Simular análise dos agentes (implementação completa seria aqui)
    with st.spinner("🤖 Agentes trabalhando..."):

        # Simular ativação dos agentes
        agents_sequence = ['orchestrator', 'data_loader',
                           'analyzer', 'visualization', 'memory']

        for i, agent in enumerate(agents_sequence):
            st.session_state.agent_status[agent] = 'active'
            time.sleep(0.5)  # Simular processamento

            # Simular progresso
            progress_bar = st.progress(0)
            for j in range(100):
                progress_bar.progress(j + 1)
                time.sleep(0.01)

            st.session_state.agent_status[agent] = 'completed'

        # Simular resultados de análise
        results = simulate_analysis_results(question, df)
        st.session_state.analysis_results = results

        # Adicionar à história
        st.session_state.analysis_history.append({
            'question': question,
            'timestamp': datetime.now(),
            'results': results
        })

    st.success("✅ Análise concluída!")
    st.rerun()


def simulate_analysis_results(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Simula resultados de análise (substituir pela implementação real)"""

    # Análise estatística básica
    numeric_columns = df.select_dtypes(include=['number']).columns

    results = {
        'question': question,
        'executive_summary': f"Análise concluída para a pergunta: '{question}'",
        'insights': [
            f"Dataset possui {len(df)} registros e {len(df.columns)} colunas",
            f"Encontradas {len(numeric_columns)} colunas numéricas",
            "Análise detalhada realizada pelos agentes especializados"
        ],
        'statistics': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        },
        'visualizations': [],
        'recommendations': [
            "Continue explorando com mais perguntas específicas",
            "Considere análises de correlação para variáveis numéricas",
            "Verifique a qualidade dos dados antes de análises avançadas"
        ]
    }

    return results


def render_analysis_results():
    """Renderiza os resultados da análise"""

    results = st.session_state.analysis_results

    # Resumo executivo
    st.subheader("📋 Resumo Executivo")
    st.info(results.get('executive_summary', 'Nenhum resumo disponível'))

    # Insights principais
    st.subheader("💡 Principais Insights")
    insights = results.get('insights', [])
    for i, insight in enumerate(insights):
        st.markdown(f"""
        <div class="insight-box">
            <strong>🔍 Insight {i+1}:</strong> {insight}
        </div>
        """, unsafe_allow_html=True)

    # Estatísticas
    if 'statistics' in results:
        st.subheader("📊 Detalhes Técnicos")
        stats = results['statistics']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📏 Total de Registros", stats.get('total_rows', 0))

        with col2:
            st.metric("📋 Colunas", stats.get('total_columns', 0))

        with col3:
            st.metric("🔢 Colunas Numéricas", stats.get('numeric_columns', 0))

        with col4:
            st.metric("❓ Valores Ausentes", stats.get('missing_values', 0))

    # Recomendações
    st.subheader("🎯 Recomendações")
    recommendations = results.get('recommendations', [])
    for rec in recommendations:
        st.markdown(f"• {rec}")


def render_data_preview(df: pd.DataFrame):
    """Renderiza preview dos dados"""

    st.subheader(f"📊 Preview: {st.session_state.current_dataset}")

    # Métricas básicas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📏 Linhas", len(df))

    with col2:
        st.metric("📋 Colunas", len(df.columns))

    with col3:
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        st.metric("🔢 Numéricas", numeric_cols)

    with col4:
        missing_pct = (df.isnull().sum().sum() /
                       (len(df) * len(df.columns))) * 100
        st.metric("❓ Ausentes (%)", f"{missing_pct:.1f}%")

    # Tabs para diferentes views
    preview_tab1, preview_tab2, preview_tab3 = st.tabs([
        "🔍 Primeiras Linhas",
        "📈 Estatísticas",
        "🏷️ Tipos de Dados"
    ])

    with preview_tab1:
        st.dataframe(df.head(100), use_container_width=True)

    with preview_tab2:
        st.write("**Estatísticas Descritivas:**")
        st.dataframe(df.describe(), use_container_width=True)

    with preview_tab3:
        col_info = pd.DataFrame({
            'Coluna': df.columns,
            'Tipo': df.dtypes,
            'Não-Nulos': df.count(),
            'Nulos': df.isnull().sum(),
            '% Nulos': (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(col_info, use_container_width=True)


def render_visualizations():
    """Renderiza visualizações"""

    st.subheader("📈 Visualizações Interativas")

    if not st.session_state.analysis_results:
        st.info("Execute uma análise primeiro para gerar visualizações")
        return

    # Placeholder para visualizações
    st.info("🚧 Visualizações serão exibidas aqui após a análise completa dos agentes")


def render_analysis_memory():
    """Renderiza memória de análises"""

    st.subheader("🧠 Histórico de Análises")

    if not st.session_state.analysis_history:
        st.info("Nenhuma análise realizada ainda")
        return

    for i, analysis in enumerate(reversed(st.session_state.analysis_history)):
        with st.expander(f"📝 Análise {len(st.session_state.analysis_history) - i}: {analysis['question'][:50]}..."):
            st.write(
                f"**📅 Data:** {analysis['timestamp'].strftime('%d/%m/%Y %H:%M:%S')}")
            st.write(f"**❓ Pergunta:** {analysis['question']}")
            st.write(
                f"**📊 Resumo:** {analysis['results'].get('executive_summary', 'Sem resumo')}")

            if st.button(f"🔄 Reexecutar", key=f"rerun_{i}"):
                st.session_state.user_question = analysis['question']
                st.rerun()


def main():
    """Função principal da aplicação"""

    # Inicializar estado da sessão
    initialize_session_state()

    # Renderizar interface
    render_header()
    render_sidebar()
    render_main_interface()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🚀 CSV Agent System v1.0 | Powered by CrewAI, Gemini & Supabase
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
