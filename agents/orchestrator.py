"""
Orchestrator Agent - Coordenador da equipe de agentes de análise de CSV
"""
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any

from ..config.prompts import ORCHESTRATOR_PROMPT
from ..config.settings import settings
from ..tools.memory_tools import get_memory_context, store_analysis_memory

class OrchestratorAgent:
    """Agente orquestrador que coordena toda a análise de CSV"""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.3
        )

        self.agent = Agent(
            role=ORCHESTRATOR_PROMPT["role"],
            goal=ORCHESTRATOR_PROMPT["goal"],
            backstory=ORCHESTRATOR_PROMPT["backstory"],
            instructions=ORCHESTRATOR_PROMPT["instructions"],
            llm=self.llm,
            tools=[get_memory_context],
            verbose=True,
            allow_delegation=True,
            max_iter=3,
            memory=True
        )

    def create_analysis_plan(self, user_question: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cria um plano de análise baseado na pergunta do usuário

        Args:
            user_question: Pergunta do usuário
            dataset_info: Informações sobre o dataset

        Returns:
            Plano de análise estruturado
        """
        # Categorizar tipo de pergunta
        question_type = self._categorize_question(user_question)

        # Identificar colunas relevantes
        relevant_columns = self._identify_relevant_columns(user_question, dataset_info)

        # Criar fluxo de execução
        execution_plan = {
            "question": user_question,
            "question_type": question_type,
            "relevant_columns": relevant_columns,
            "dataset_info": dataset_info,
            "steps": self._create_execution_steps(question_type, relevant_columns),
            "expected_outputs": self._define_expected_outputs(question_type),
            "visualization_needs": self._identify_visualization_needs(question_type)
        }

        return execution_plan

    def _categorize_question(self, question: str) -> str:
        """Categoriza o tipo de pergunta do usuário"""
        question_lower = question.lower()

        # Palavras-chave para diferentes tipos de análise
        keywords = {
            "descriptive": ["descreva", "resumo", "estatísticas", "média", "mediana", "distribuição"],
            "correlation": ["correlação", "relacionamento", "associação", "dependência"],
            "comparison": ["compare", "diferença", "maior", "menor", "versus", "vs"],
            "trend": ["tendência", "evolução", "crescimento", "temporal", "tempo"],
            "outlier": ["outliers", "anômalos", "extremos", "discrepantes"],
            "categorical": ["categorias", "grupos", "segmentos", "classes"],
            "predictive": ["prever", "predição", "futuro", "modelo"],
            "custom": ["gráfico", "visualização", "plot", "chart"]
        }

        # Contar ocorrências de palavras-chave
        scores = {}
        for category, words in keywords.items():
            scores[category] = sum(1 for word in words if word in question_lower)

        # Retornar categoria com maior score
        if max(scores.values()) == 0:
            return "exploratory"  # Análise exploratória geral

        return max(scores, key=scores.get)

    def _identify_relevant_columns(self, question: str, dataset_info: Dict[str, Any]) -> List[str]:
        """Identifica colunas relevantes para a pergunta"""
        question_lower = question.lower()
        columns = dataset_info.get("columns", [])

        relevant_columns = []

        # Buscar menções diretas de colunas na pergunta
        for col in columns:
            if col.lower() in question_lower:
                relevant_columns.append(col)

        # Se não encontrou colunas específicas, usar heurísticas
        if not relevant_columns:
            column_types = dataset_info.get("data_types_analysis", {})

            # Para análises de correlação, incluir colunas numéricas
            if "correlação" in question_lower or "relacionamento" in question_lower:
                relevant_columns = [col for col, info in column_types.items() 
                                 if info.get("semantic_type") == "numerical"]

            # Para análises temporais, incluir colunas de data
            elif "tempo" in question_lower or "temporal" in question_lower:
                relevant_columns = [col for col, info in column_types.items() 
                                 if info.get("semantic_type") == "datetime"]

            # Para análises categóricas
            elif "categoria" in question_lower or "grupo" in question_lower:
                relevant_columns = [col for col, info in column_types.items() 
                                 if info.get("semantic_type") == "categorical"]

        # Se ainda não encontrou, usar todas as colunas (análise exploratória)
        if not relevant_columns:
            relevant_columns = columns[:10]  # Limitar a 10 colunas

        return relevant_columns

    def _create_execution_steps(self, question_type: str, relevant_columns: List[str]) -> List[Dict[str, Any]]:
        """Cria as etapas de execução baseada no tipo de pergunta"""

        steps = []

        # Sempre começar com carregamento e validação
        steps.append({
            "agent": "DataLoader",
            "task": "load_and_validate",
            "description": "Carregar e validar o dataset CSV",
            "priority": 1
        })

        # Etapas específicas por tipo de pergunta
        if question_type == "descriptive":
            steps.extend([
                {
                    "agent": "Analyzer",
                    "task": "descriptive_analysis",
                    "description": "Realizar análise descritiva completa",
                    "columns": relevant_columns,
                    "priority": 2
                },
                {
                    "agent": "Visualization",
                    "task": "create_distribution_plots",
                    "description": "Criar gráficos de distribuição",
                    "columns": relevant_columns,
                    "priority": 3
                }
            ])

        elif question_type == "correlation":
            steps.extend([
                {
                    "agent": "Analyzer", 
                    "task": "correlation_analysis",
                    "description": "Analisar correlações entre variáveis",
                    "columns": relevant_columns,
                    "priority": 2
                },
                {
                    "agent": "Visualization",
                    "task": "create_correlation_matrix",
                    "description": "Criar matriz de correlação",
                    "columns": relevant_columns,
                    "priority": 3
                }
            ])

        elif question_type == "comparison":
            steps.extend([
                {
                    "agent": "Analyzer",
                    "task": "comparative_analysis", 
                    "description": "Realizar análise comparativa",
                    "columns": relevant_columns,
                    "priority": 2
                },
                {
                    "agent": "Visualization",
                    "task": "create_comparison_plots",
                    "description": "Criar gráficos comparativos",
                    "columns": relevant_columns,
                    "priority": 3
                }
            ])

        elif question_type == "trend":
            steps.extend([
                {
                    "agent": "Analyzer",
                    "task": "trend_analysis",
                    "description": "Analisar tendências temporais",
                    "columns": relevant_columns,
                    "priority": 2
                },
                {
                    "agent": "Visualization",
                    "task": "create_time_series",
                    "description": "Criar gráficos de série temporal",
                    "columns": relevant_columns,
                    "priority": 3
                }
            ])

        elif question_type == "outlier":
            steps.extend([
                {
                    "agent": "Analyzer",
                    "task": "outlier_detection",
                    "description": "Detectar e analisar outliers",
                    "columns": relevant_columns,
                    "priority": 2
                },
                {
                    "agent": "Visualization",
                    "task": "create_outlier_plots",
                    "description": "Criar gráficos de outliers (boxplots)",
                    "columns": relevant_columns,
                    "priority": 3
                }
            ])

        else:  # exploratory ou outros
            steps.extend([
                {
                    "agent": "Analyzer",
                    "task": "exploratory_analysis",
                    "description": "Realizar análise exploratória completa",
                    "columns": relevant_columns,
                    "priority": 2
                },
                {
                    "agent": "Visualization",
                    "task": "create_overview_dashboard",
                    "description": "Criar dashboard overview",
                    "columns": relevant_columns,
                    "priority": 3
                }
            ])

        # Sempre finalizar com armazenamento na memória
        steps.append({
            "agent": "Memory",
            "task": "store_results",
            "description": "Armazenar resultados na memória semântica",
            "priority": 4
        })

        return steps

    def _define_expected_outputs(self, question_type: str) -> Dict[str, str]:
        """Define os outputs esperados para cada tipo de pergunta"""

        outputs = {
            "descriptive": {
                "summary": "Resumo estatístico completo",
                "insights": "Lista de insights principais",
                "recommendations": "Recomendações de próximos passos"
            },
            "correlation": {
                "correlation_matrix": "Matriz de correlação",
                "significant_correlations": "Correlações significativas encontradas",
                "interpretation": "Interpretação das correlações"
            },
            "comparison": {
                "comparison_results": "Resultados da comparação",
                "statistical_tests": "Testes estatísticos realizados",
                "conclusions": "Conclusões da comparação"
            },
            "trend": {
                "trend_analysis": "Análise de tendências",
                "patterns": "Padrões temporais identificados",
                "forecasting": "Projeções se aplicável"
            },
            "outlier": {
                "outliers_detected": "Lista de outliers detectados",
                "outlier_analysis": "Análise dos outliers",
                "recommendations": "Recomendações de tratamento"
            }
        }

        return outputs.get(question_type, {
            "analysis": "Análise exploratória geral",
            "insights": "Insights descobertos",
            "visualizations": "Visualizações criadas"
        })

    def _identify_visualization_needs(self, question_type: str) -> List[str]:
        """Identifica quais visualizações são necessárias"""

        viz_mapping = {
            "descriptive": ["histogram", "boxplot", "summary_table"],
            "correlation": ["correlation_heatmap", "scatter_plots"],
            "comparison": ["bar_chart", "violin_plot", "comparison_table"],
            "trend": ["time_series", "trend_lines"],
            "outlier": ["boxplot", "scatter_plot", "outlier_table"],
            "categorical": ["bar_chart", "pie_chart", "count_plot"]
        }

        return viz_mapping.get(question_type, ["histogram", "scatter_plot", "summary_table"])

    def consolidate_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolida todos os resultados em uma resposta final estruturada

        Args:
            analysis_results: Resultados de todos os agentes

        Returns:
            Resposta consolidada para o usuário
        """

        consolidated = {
            "executive_summary": self._create_executive_summary(analysis_results),
            "technical_details": self._extract_technical_details(analysis_results),
            "visualizations": self._collect_visualizations(analysis_results),
            "memory_context": analysis_results.get("memory_context", {}),
            "recommendations": self._generate_recommendations(analysis_results),
            "metadata": {
                "analysis_timestamp": analysis_results.get("timestamp"),
                "dataset_info": analysis_results.get("dataset_info", {}),
                "question": analysis_results.get("original_question"),
                "agents_involved": list(analysis_results.keys())
            }
        }

        return consolidated

    def _create_executive_summary(self, results: Dict[str, Any]) -> str:
        """Cria resumo executivo dos resultados"""

        summary_parts = []

        # Adicionar insights principais
        if "analyzer_results" in results:
            analyzer_data = results["analyzer_results"]
            if "insights" in analyzer_data:
                summary_parts.append("📊 Principais descobertas:")
                summary_parts.extend([f"• {insight}" for insight in analyzer_data["insights"][:3]])

        # Adicionar informações de visualização
        if "visualization_results" in results:
            viz_data = results["visualization_results"]
            if "charts_created" in viz_data:
                summary_parts.append(f"\n📈 Visualizações geradas: {len(viz_data['charts_created'])} gráficos interativos")

        # Adicionar contexto de memória se disponível
        if "memory_context" in results and results["memory_context"].get("has_previous_context"):
            summary_parts.append("\n🧠 Análises anteriores relacionadas foram consideradas")

        return "\n".join(summary_parts) if summary_parts else "Análise concluída com sucesso"

    def _extract_technical_details(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai detalhes técnicos dos resultados"""

        technical = {}

        if "data_loader_results" in results:
            technical["data_quality"] = results["data_loader_results"].get("quality_report", {})
            technical["data_structure"] = results["data_loader_results"].get("data_types_analysis", {})

        if "analyzer_results" in results:
            technical["statistics"] = results["analyzer_results"].get("statistics", {})
            technical["correlations"] = results["analyzer_results"].get("correlations", {})
            technical["outliers"] = results["analyzer_results"].get("outliers", {})

        return technical

    def _collect_visualizations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Coleta todas as visualizações criadas"""

        visualizations = []

        if "visualization_results" in results:
            viz_data = results["visualization_results"]
            if "charts" in viz_data:
                visualizations.extend(viz_data["charts"])

        return visualizations

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas nos resultados"""

        recommendations = []

        # Recomendações baseadas na qualidade dos dados
        if "data_loader_results" in results:
            quality = results["data_loader_results"].get("quality_report", {})

            if quality.get("null_percentage", 0) > 10:
                recommendations.append("Considerar estratégias de tratamento de valores ausentes")

            if quality.get("duplicate_percentage", 0) > 5:
                recommendations.append("Investigar e possivelmente remover registros duplicados")

            if quality.get("outliers_detected", 0) > 0:
                recommendations.append("Analisar outliers para determinar se são válidos ou erros")

        # Recomendações baseadas na análise
        if "analyzer_results" in results:
            analyzer_data = results["analyzer_results"]

            if "strong_correlations" in analyzer_data:
                recommendations.append("Explorar as correlações fortes identificadas para insights de negócio")

            if "temporal_patterns" in analyzer_data:
                recommendations.append("Considerar análise de séries temporais mais avançada")

        return recommendations if recommendations else ["Continue explorando os dados com novas perguntas"]


def create_orchestrator_agent() -> Agent:
    """Factory function para criar o agente orquestrador"""
    orchestrator = OrchestratorAgent()
    return orchestrator.agent
