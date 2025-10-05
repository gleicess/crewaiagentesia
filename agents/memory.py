"""
Memory Agent - Gestor de memória semântica integrado ao Supabase Vector
"""
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json

from ..config.prompts import MEMORY_PROMPT
from ..config.settings import settings
from ..tools.memory_tools import (
    store_analysis_memory,
    search_analysis_memory,
    get_memory_context,
    update_analysis_memory,
    initialize_memory_manager
)

logger = logging.getLogger(__name__)

class MemoryAgent:
    """Agente especializado em gestão de memória semântica"""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.1  # Muito baixa para consistência na memória
        )

        self.agent = Agent(
            role=MEMORY_PROMPT["role"],
            goal=MEMORY_PROMPT["goal"],
            backstory=MEMORY_PROMPT["backstory"],
            instructions=MEMORY_PROMPT["instructions"],
            llm=self.llm,
            tools=[
                store_analysis_memory,
                search_analysis_memory,
                get_memory_context,
                update_analysis_memory
            ],
            verbose=True,
            memory=True,
            max_iter=2
        )

        # Inicializar gerenciador de memória
        self._initialize_memory_system()

    def _initialize_memory_system(self):
        """Inicializa o sistema de memória"""
        try:
            initialize_memory_manager(
                supabase_url=settings.supabase_url,
                supabase_key=settings.supabase_anon_key,
                openai_api_key=settings.openai_api_key
            )
            logger.info("Sistema de memória inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro na inicialização da memória: {e}")

    def store_comprehensive_analysis(self, analysis_results: Dict[str, Any], 
                                   dataset_name: str, user_question: str) -> Dict[str, Any]:
        """
        Armazena resultados completos de análise na memória semântica

        Args:
            analysis_results: Resultados consolidados da análise
            dataset_name: Nome do dataset analisado
            user_question: Pergunta original do usuário

        Returns:
            Dict com resultado do armazenamento
        """
        try:
            # Extrair informações principais para armazenamento
            main_insights = self._extract_key_insights(analysis_results)
            statistics_summary = self._summarize_statistics(analysis_results)
            visualizations_info = self._summarize_visualizations(analysis_results)

            # Criar conteúdo principal para embedding
            content = self._create_memory_content(
                user_question, main_insights, statistics_summary, visualizations_info
            )

            # Determinar tipo de análise
            analysis_type = self._classify_analysis_type(user_question, analysis_results)

            # Criar tags para categorização
            tags = self._generate_tags(dataset_name, analysis_type, analysis_results)

            # Armazenar na memória
            storage_result = store_analysis_memory(
                content=content,
                analysis_type=analysis_type,
                dataset_name=dataset_name,
                question=user_question,
                insights=main_insights,
                statistics=statistics_summary,
                tags=tags
            )

            if storage_result.get("success"):
                # Também armazenar visualizações separadamente se existirem
                if visualizations_info:
                    viz_storage = self._store_visualization_memory(
                        visualizations_info, dataset_name, user_question, 
                        storage_result["memory_id"]
                    )

                return {
                    "success": True,
                    "memory_id": storage_result["memory_id"],
                    "content_stored": len(content),
                    "insights_count": len(main_insights),
                    "tags": tags,
                    "analysis_type": analysis_type,
                    "timestamp": storage_result["timestamp"]
                }
            else:
                return storage_result

        except Exception as e:
            logger.error(f"Erro no armazenamento da análise: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def retrieve_relevant_context(self, current_question: str, dataset_name: str,
                                max_memories: int = 5) -> Dict[str, Any]:
        """
        Recupera contexto relevante para a pergunta atual

        Args:
            current_question: Pergunta atual do usuário
            dataset_name: Nome do dataset atual
            max_memories: Máximo de memórias para recuperar

        Returns:
            Dict com contexto relevante recuperado
        """
        try:
            # Buscar contexto usando ferramenta de memória
            context_result = get_memory_context(
                current_question=current_question,
                dataset_name=dataset_name,
                max_context_memories=max_memories
            )

            if context_result.get("success"):
                context = context_result["context"]

                # Enriquecer contexto com análise semântica
                enriched_context = self._enrich_context(context, current_question)

                return {
                    "success": True,
                    "has_context": context_result["has_previous_context"],
                    "context": enriched_context,
                    "related_analyses_count": len(context.get("related_analyses", [])),
                    "dataset_history_count": len(context.get("dataset_history", [])),
                    "context_summary": context.get("context_summary", ""),
                    "recommendations": self._generate_context_recommendations(enriched_context)
                }
            else:
                return context_result

        except Exception as e:
            logger.error(f"Erro na recuperação de contexto: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def search_similar_analyses(self, query: str, analysis_type: Optional[str] = None,
                              dataset_name: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Busca análises similares na memória

        Args:
            query: Consulta de busca
            analysis_type: Filtro por tipo de análise
            dataset_name: Filtro por dataset
            limit: Número máximo de resultados

        Returns:
            Dict com análises similares encontradas
        """
        try:
            search_result = search_analysis_memory(
                query=query,
                analysis_type=analysis_type,
                dataset_name=dataset_name,
                limit=limit
            )

            if search_result.get("success"):
                # Processar e enriquecer resultados
                processed_results = self._process_search_results(search_result["results"])

                return {
                    "success": True,
                    "query": query,
                    "results_count": search_result["results_count"],
                    "results": processed_results,
                    "search_insights": self._generate_search_insights(processed_results),
                    "related_datasets": self._extract_related_datasets(processed_results),
                    "common_patterns": self._identify_common_patterns(processed_results)
                }
            else:
                return search_result

        except Exception as e:
            logger.error(f"Erro na busca de análises similares: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def update_memory_with_feedback(self, memory_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Atualiza memória existente com feedback ou novas descobertas

        Args:
            memory_id: ID da memória a ser atualizada
            feedback: Feedback ou novas informações

        Returns:
            Dict com resultado da atualização
        """
        try:
            # Processar feedback em insights atualizados
            additional_insights = feedback.get("additional_insights", [])
            updated_statistics = feedback.get("updated_statistics", {})
            new_tags = feedback.get("new_tags", [])

            # Atualizar memória
            update_result = update_analysis_memory(
                memory_id=memory_id,
                additional_insights=additional_insights,
                updated_statistics=updated_statistics,
                new_tags=new_tags
            )

            return update_result

        except Exception as e:
            logger.error(f"Erro na atualização da memória: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_memory_analytics(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtém analytics da memória (estatísticas de uso)

        Args:
            dataset_name: Filtro opcional por dataset

        Returns:
            Dict com analytics da memória
        """
        try:
            # Buscar todas as análises
            search_result = search_analysis_memory(
                query="*",  # Busca geral
                dataset_name=dataset_name,
                limit=1000  # Limite alto para analytics
            )

            if search_result.get("success"):
                results = search_result["results"]

                analytics = {
                    "total_analyses": len(results),
                    "analysis_types": self._count_analysis_types(results),
                    "datasets_analyzed": self._count_datasets(results),
                    "temporal_distribution": self._analyze_temporal_distribution(results),
                    "most_common_questions": self._extract_common_questions(results),
                    "insight_patterns": self._analyze_insight_patterns(results),
                    "memory_health": self._assess_memory_health(results)
                }

                return {
                    "success": True,
                    "analytics": analytics,
                    "generated_at": datetime.now().isoformat()
                }
            else:
                return search_result

        except Exception as e:
            logger.error(f"Erro nos analytics de memória: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    # Métodos auxiliares privados

    def _extract_key_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extrai insights principais dos resultados da análise"""
        insights = []

        # Insights do analisador
        if "insights" in analysis_results:
            insights.extend(analysis_results["insights"])

        # Insights das visualizações
        if "visualizations" in analysis_results:
            for viz in analysis_results["visualizations"]:
                if isinstance(viz, dict) and "insights" in viz:
                    insights.extend(viz["insights"])

        # Insights do resumo executivo
        if "executive_summary" in analysis_results:
            summary = analysis_results["executive_summary"]
            if isinstance(summary, str):
                insights.append(summary)

        return list(set(insights))  # Remove duplicatas

    def _summarize_statistics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Resume estatísticas principais"""
        summary = {}

        if "technical_details" in analysis_results:
            tech_details = analysis_results["technical_details"]

            # Estatísticas descritivas
            if "statistics" in tech_details:
                stats = tech_details["statistics"]
                # Pegar apenas estatísticas principais para economizar espaço
                summary["descriptive_stats"] = {
                    col: {k: v for k, v in col_stats.items() if k in ["mean", "median", "std", "min", "max"]}
                    for col, col_stats in stats.items()
                    if isinstance(col_stats, dict)
                }

            # Correlações principais
            if "correlations" in tech_details:
                corr_data = tech_details["correlations"]
                if "strong_correlations" in corr_data:
                    summary["key_correlations"] = corr_data["strong_correlations"][:5]  # Top 5

            # Outliers summary
            if "outliers" in tech_details:
                outlier_data = tech_details["outliers"]
                summary["outlier_summary"] = {
                    col: info.get("iqr_method", {}).get("outlier_percentage", 0)
                    for col, info in outlier_data.items()
                    if isinstance(info, dict)
                }

        return summary

    def _summarize_visualizations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resume informações das visualizações"""
        viz_summary = []

        if "visualizations" in analysis_results:
            for viz in analysis_results["visualizations"]:
                if isinstance(viz, dict):
                    viz_info = {
                        "type": viz.get("type", "unknown"),
                        "title": viz.get("title", ""),
                        "insights": viz.get("insights", [])[:3],  # Top 3 insights
                        "columns_analyzed": viz.get("columns_analyzed", [])
                    }
                    viz_summary.append(viz_info)

        return viz_summary

    def _create_memory_content(self, question: str, insights: List[str], 
                             statistics: Dict, visualizations: List) -> str:
        """Cria conteúdo principal para embedding"""
        content_parts = [
            f"Pergunta: {question}",
            "",
            "Principais descobertas:"
        ]

        for i, insight in enumerate(insights[:10], 1):  # Top 10 insights
            content_parts.append(f"{i}. {insight}")

        if statistics:
            content_parts.extend([
                "",
                "Estatísticas principais:",
                json.dumps(statistics, indent=2)[:500]  # Limitar tamanho
            ])

        if visualizations:
            content_parts.extend([
                "",
                "Visualizações criadas:",
                ", ".join([viz.get("type", "") for viz in visualizations])
            ])

        return "\n".join(content_parts)

    def _classify_analysis_type(self, question: str, results: Dict) -> str:
        """Classifica o tipo de análise baseado na pergunta e resultados"""
        question_lower = question.lower()

        if "correlação" in question_lower or "relacionamento" in question_lower:
            return "correlation"
        elif "outlier" in question_lower or "anômalo" in question_lower:
            return "outlier_detection"
        elif "tendência" in question_lower or "temporal" in question_lower:
            return "trend_analysis"
        elif "distribuição" in question_lower or "histograma" in question_lower:
            return "distribution_analysis"
        elif "compare" in question_lower or "diferença" in question_lower:
            return "comparison"
        elif "cluster" in question_lower or "grupo" in question_lower:
            return "clustering"
        else:
            return "exploratory"

    def _generate_tags(self, dataset_name: str, analysis_type: str, results: Dict) -> List[str]:
        """Gera tags para categorização"""
        tags = [analysis_type, dataset_name, "csv_analysis"]

        # Tags baseadas nos resultados
        if "technical_details" in results:
            tech_details = results["technical_details"]

            if "correlations" in tech_details:
                strong_corrs = tech_details["correlations"].get("strong_correlations", [])
                if strong_corrs:
                    tags.append("strong_correlations")

            if "outliers" in tech_details:
                tags.append("outlier_analysis")

        if "visualizations" in results:
            viz_types = [viz.get("type", "") for viz in results["visualizations"]]
            tags.extend([f"viz_{vtype}" for vtype in set(viz_types) if vtype])

        return list(set(tags))  # Remove duplicatas

    def _store_visualization_memory(self, viz_info: List[Dict], dataset_name: str,
                                  question: str, parent_memory_id: str) -> Dict[str, Any]:
        """Armazena informações de visualização separadamente"""
        try:
            viz_content = f"Visualizações para: {question}\n"
            viz_content += "\n".join([
                f"- {viz['type']}: {viz['title']}" 
                for viz in viz_info if 'type' in viz and 'title' in viz
            ])

            return store_analysis_memory(
                content=viz_content,
                analysis_type="visualization",
                dataset_name=dataset_name,
                question=question,
                insights=[],
                tags=["visualization", dataset_name, "supplementary"],
                namespace="csv_visualization"
            )
        except Exception as e:
            logger.warning(f"Erro ao armazenar memória de visualização: {e}")
            return {"success": False, "error": str(e)}

    def _enrich_context(self, context: Dict, current_question: str) -> Dict[str, Any]:
        """Enriquece contexto com análise semântica"""
        enriched = context.copy()

        # Analisar relevância das análises relacionadas
        related_analyses = context.get("related_analyses", [])
        if related_analyses:
            enriched["relevance_scores"] = self._calculate_relevance_scores(
                related_analyses, current_question
            )

        # Sugerir conexões entre análises passadas
        enriched["suggested_connections"] = self._suggest_analysis_connections(related_analyses)

        return enriched

    def _calculate_relevance_scores(self, analyses: List[Dict], question: str) -> List[float]:
        """Calcula scores de relevância simplificados"""
        scores = []
        question_words = set(question.lower().split())

        for analysis in analyses:
            analysis_question = analysis.get("question", "").lower()
            analysis_words = set(analysis_question.split())

            # Similaridade baseada em palavras em comum
            common_words = question_words.intersection(analysis_words)
            similarity = len(common_words) / max(len(question_words), 1)
            scores.append(similarity)

        return scores

    def _suggest_analysis_connections(self, analyses: List[Dict]) -> List[str]:
        """Sugere conexões entre análises"""
        connections = []

        analysis_types = [a.get("analysis_type", "") for a in analyses]
        type_counts = {}
        for atype in analysis_types:
            type_counts[atype] = type_counts.get(atype, 0) + 1

        # Sugerir padrões
        for atype, count in type_counts.items():
            if count > 1:
                connections.append(f"Múltiplas análises de {atype} já realizadas")

        return connections

    def _process_search_results(self, results: List[Dict]) -> List[Dict]:
        """Processa e enriquece resultados da busca"""
        processed = []

        for result in results:
            processed_result = result.copy()

            # Adicionar score de relevância processado
            processed_result["relevance_category"] = self._categorize_relevance(
                result.get("relevance_score", 0)
            )

            # Extrair palavras-chave
            processed_result["keywords"] = self._extract_keywords(
                result.get("content", "")
            )

            processed.append(processed_result)

        return processed

    def _categorize_relevance(self, score: float) -> str:
        """Categoriza score de relevância"""
        if score > 0.8:
            return "Muito relevante"
        elif score > 0.6:
            return "Relevante"
        elif score > 0.4:
            return "Moderadamente relevante"
        else:
            return "Pouco relevante"

    def _extract_keywords(self, content: str) -> List[str]:
        """Extrai palavras-chave do conteúdo"""
        # Implementação simplificada
        words = content.lower().split()
        keywords = [w for w in words if len(w) > 4 and w.isalpha()]
        return list(set(keywords))[:10]  # Top 10 palavras únicas

    def _generate_search_insights(self, results: List[Dict]) -> List[str]:
        """Gera insights dos resultados da busca"""
        insights = []

        if not results:
            return ["Nenhuma análise similar encontrada"]

        insights.append(f"Encontradas {len(results)} análises relacionadas")

        # Análise de tipos
        types = [r.get("analysis_type", "") for r in results]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1

        most_common = max(type_counts, key=type_counts.get) if type_counts else None
        if most_common:
            insights.append(f"Tipo mais comum: {most_common} ({type_counts[most_common]} análises)")

        return insights

    def _extract_related_datasets(self, results: List[Dict]) -> List[str]:
        """Extrai datasets relacionados dos resultados"""
        datasets = list(set([r.get("dataset_name", "") for r in results if r.get("dataset_name")]))
        return datasets

    def _identify_common_patterns(self, results: List[Dict]) -> List[str]:
        """Identifica padrões comuns nos resultados"""
        patterns = []

        # Padrões de insights
        all_insights = []
        for result in results:
            insights = result.get("insights", [])
            all_insights.extend(insights)

        # Encontrar insights recorrentes (simplificado)
        insight_words = {}
        for insight in all_insights:
            words = insight.lower().split()
            for word in words:
                if len(word) > 4:
                    insight_words[word] = insight_words.get(word, 0) + 1

        common_words = [word for word, count in insight_words.items() if count > 2]
        if common_words:
            patterns.append(f"Palavras recorrentes: {', '.join(common_words[:5])}")

        return patterns

    def _generate_context_recommendations(self, context: Dict) -> List[str]:
        """Gera recomendações baseadas no contexto"""
        recommendations = []

        if context.get("has_previous_context"):
            recommendations.append("Considere os insights de análises anteriores")
            recommendations.append("Compare resultados com análises históricas")

        related_count = len(context.get("related_analyses", []))
        if related_count > 0:
            recommendations.append(f"Revisar {related_count} análises relacionadas disponíveis")

        if not context.get("has_previous_context"):
            recommendations.append("Esta é uma nova análise - considere criar baseline")

        return recommendations

    # Métodos para analytics

    def _count_analysis_types(self, results: List[Dict]) -> Dict[str, int]:
        """Conta tipos de análise"""
        counts = {}
        for result in results:
            atype = result.get("analysis_type", "unknown")
            counts[atype] = counts.get(atype, 0) + 1
        return counts

    def _count_datasets(self, results: List[Dict]) -> Dict[str, int]:
        """Conta datasets analisados"""
        counts = {}
        for result in results:
            dataset = result.get("dataset_name", "unknown")
            counts[dataset] = counts.get(dataset, 0) + 1
        return counts

    def _analyze_temporal_distribution(self, results: List[Dict]) -> Dict[str, Any]:
        """Analisa distribuição temporal das análises"""
        # Implementação simplificada
        return {
            "total_span": f"{len(results)} análises registradas",
            "frequency": "Análise de padrões temporais não implementada"
        }

    def _extract_common_questions(self, results: List[Dict]) -> List[str]:
        """Extrai perguntas mais comuns"""
        questions = [r.get("question", "") for r in results if r.get("question")]
        # Retornar amostra das primeiras perguntas
        return questions[:10]

    def _analyze_insight_patterns(self, results: List[Dict]) -> Dict[str, Any]:
        """Analisa padrões nos insights"""
        return {
            "total_insights": sum(len(r.get("insights", [])) for r in results),
            "pattern_analysis": "Análise de padrões de insights não implementada"
        }

    def _assess_memory_health(self, results: List[Dict]) -> Dict[str, Any]:
        """Avalia saúde da memória"""
        return {
            "total_memories": len(results),
            "health_status": "good" if len(results) > 0 else "no_data",
            "diversity_score": len(set(r.get("analysis_type", "") for r in results))
        }


def create_memory_agent() -> Agent:
    """Factory function para criar o agente Memory"""
    memory = MemoryAgent()
    return memory.agent
