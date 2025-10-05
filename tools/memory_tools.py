"""
Ferramentas para memória semântica usando Supabase Vector
"""
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import json
import logging
import asyncio
from dataclasses import dataclass

from supabase import create_client, Client
import openai
from crewai_tools import tool

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Entrada de memória"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
    timestamp: datetime
    tags: List[str]
    namespace: str

class SupabaseMemoryManager:
    """Gerenciador de memória usando Supabase Vector"""

    def __init__(self, supabase_url: str, supabase_key: str, openai_api_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.table_name = "agent_memories"
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Garante que a tabela de memórias existe"""
        try:
            # Verificar se a tabela existe
            result = self.client.table(self.table_name).select("id").limit(1).execute()
        except Exception as e:
            logger.info(f"Tabela {self.table_name} não existe, será criada automaticamente")

    def _generate_embedding(self, text: str) -> List[float]:
        """Gera embedding para um texto usando OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            return []

    def store_memory(
        self, 
        content: str, 
        metadata: Dict[str, Any] = None,
        tags: List[str] = None,
        namespace: str = "default"
    ) -> str:
        """
        Armazena uma nova memória

        Args:
            content: Conteúdo da memória
            metadata: Metadados adicionais
            tags: Tags para categorização
            namespace: Namespace da memória

        Returns:
            ID da memória armazenada
        """
        memory_id = str(uuid.uuid4())
        embedding = self._generate_embedding(content)

        memory_data = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "embedding": embedding,
            "timestamp": datetime.utcnow().isoformat(),
            "tags": tags or [],
            "namespace": namespace
        }

        try:
            result = self.client.table(self.table_name).insert(memory_data).execute()
            logger.info(f"Memória armazenada com ID: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Erro ao armazenar memória: {e}")
            raise

    def search_memories(
        self, 
        query: str, 
        namespace: str = "default",
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[MemoryEntry]:
        """
        Busca memórias similares usando busca vetorial

        Args:
            query: Consulta de busca
            namespace: Namespace para buscar
            limit: Número máximo de resultados
            similarity_threshold: Limite de similaridade

        Returns:
            Lista de memórias similares
        """
        query_embedding = self._generate_embedding(query)

        try:
            # Usar função de similaridade do pgvector
            result = self.client.rpc(
                "match_memories",
                {
                    "query_embedding": query_embedding,
                    "match_namespace": namespace,
                    "match_count": limit,
                    "similarity_threshold": similarity_threshold
                }
            ).execute()

            memories = []
            for row in result.data:
                memory = MemoryEntry(
                    id=row["id"],
                    content=row["content"],
                    metadata=row["metadata"],
                    embedding=row["embedding"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    tags=row["tags"],
                    namespace=row["namespace"]
                )
                memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Erro na busca de memórias: {e}")
            return []

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Recupera uma memória específica por ID"""
        try:
            result = self.client.table(self.table_name).select("*").eq("id", memory_id).single().execute()

            if result.data:
                row = result.data
                return MemoryEntry(
                    id=row["id"],
                    content=row["content"],
                    metadata=row["metadata"],
                    embedding=row["embedding"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    tags=row["tags"],
                    namespace=row["namespace"]
                )
            return None
        except Exception as e:
            logger.error(f"Erro ao recuperar memória {memory_id}: {e}")
            return None

    def get_memories_by_tags(self, tags: List[str], namespace: str = "default") -> List[MemoryEntry]:
        """Recupera memórias por tags"""
        try:
            result = self.client.table(self.table_name).select("*").eq("namespace", namespace).contains("tags", tags).execute()

            memories = []
            for row in result.data:
                memory = MemoryEntry(
                    id=row["id"],
                    content=row["content"],
                    metadata=row["metadata"],
                    embedding=row["embedding"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    tags=row["tags"],
                    namespace=row["namespace"]
                )
                memories.append(memory)

            return memories
        except Exception as e:
            logger.error(f"Erro ao buscar memórias por tags: {e}")
            return []


# Instância global do gerenciador (será inicializada na configuração)
memory_manager: Optional[SupabaseMemoryManager] = None


def initialize_memory_manager(supabase_url: str, supabase_key: str, openai_api_key: str):
    """Inicializa o gerenciador de memória"""
    global memory_manager
    memory_manager = SupabaseMemoryManager(supabase_url, supabase_key, openai_api_key)


@tool("store_analysis_memory")
def store_analysis_memory(
    content: str,
    analysis_type: str,
    dataset_name: str,
    question: str,
    insights: List[str] = None,
    statistics: Dict[str, Any] = None,
    tags: List[str] = None,
    namespace: str = "csv_analysis"
) -> Dict[str, Any]:
    """
    Armazena resultado de análise na memória semântica.

    Args:
        content: Conteúdo principal da análise
        analysis_type: Tipo de análise (histogram, correlation, etc.)
        dataset_name: Nome do dataset analisado
        question: Pergunta original do usuário
        insights: Lista de insights descobertos
        statistics: Estatísticas relevantes
        tags: Tags para categorização
        namespace: Namespace da memória

    Returns:
        Dict com resultado da operação
    """
    if memory_manager is None:
        return {"error": "Memory manager não inicializado"}

    try:
        metadata = {
            "analysis_type": analysis_type,
            "dataset_name": dataset_name,
            "question": question,
            "insights": insights or [],
            "statistics": statistics or {},
            "created_at": datetime.utcnow().isoformat()
        }

        # Incluir tags padrão
        default_tags = [analysis_type, "csv_analysis", dataset_name]
        all_tags = list(set(default_tags + (tags or [])))

        memory_id = memory_manager.store_memory(
            content=content,
            metadata=metadata,
            tags=all_tags,
            namespace=namespace
        )

        return {
            "success": True,
            "memory_id": memory_id,
            "stored_content": content[:100] + "..." if len(content) > 100 else content,
            "tags": all_tags,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Erro ao armazenar análise: {e}")
        return {"error": str(e)}


@tool("search_analysis_memory")
def search_analysis_memory(
    query: str,
    analysis_type: Optional[str] = None,
    dataset_name: Optional[str] = None,
    limit: int = 5,
    similarity_threshold: float = 0.7,
    namespace: str = "csv_analysis"
) -> Dict[str, Any]:
    """
    Busca análises anteriores na memória semântica.

    Args:
        query: Consulta de busca
        analysis_type: Filtrar por tipo de análise
        dataset_name: Filtrar por dataset
        limit: Número máximo de resultados
        similarity_threshold: Limite de similaridade
        namespace: Namespace para buscar

    Returns:
        Dict com resultados da busca
    """
    if memory_manager is None:
        return {"error": "Memory manager não inicializado"}

    try:
        # Buscar memórias similares
        memories = memory_manager.search_memories(
            query=query,
            namespace=namespace,
            limit=limit,
            similarity_threshold=similarity_threshold
        )

        # Filtrar por critérios adicionais se especificados
        filtered_memories = []
        for memory in memories:
            include = True

            if analysis_type and memory.metadata.get("analysis_type") != analysis_type:
                include = False

            if dataset_name and memory.metadata.get("dataset_name") != dataset_name:
                include = False

            if include:
                filtered_memories.append(memory)

        # Formatear resultados
        results = []
        for memory in filtered_memories:
            result = {
                "memory_id": memory.id,
                "content": memory.content,
                "analysis_type": memory.metadata.get("analysis_type"),
                "dataset_name": memory.metadata.get("dataset_name"),
                "question": memory.metadata.get("question"),
                "insights": memory.metadata.get("insights", []),
                "statistics": memory.metadata.get("statistics", {}),
                "tags": memory.tags,
                "timestamp": memory.timestamp.isoformat(),
                "relevance_score": 1.0  # Seria calculado pela busca vetorial
            }
            results.append(result)

        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results,
            "search_criteria": {
                "analysis_type": analysis_type,
                "dataset_name": dataset_name,
                "similarity_threshold": similarity_threshold
            }
        }

    except Exception as e:
        logger.error(f"Erro na busca de memórias: {e}")
        return {"error": str(e)}


@tool("get_memory_context")
def get_memory_context(
    current_question: str,
    dataset_name: str,
    max_context_memories: int = 3,
    namespace: str = "csv_analysis"
) -> Dict[str, Any]:
    """
    Recupera contexto relevante para a questão atual.

    Args:
        current_question: Pergunta atual do usuário
        dataset_name: Nome do dataset atual
        max_context_memories: Máximo de memórias para contexto
        namespace: Namespace para buscar

    Returns:
        Dict com contexto relevante
    """
    if memory_manager is None:
        return {"error": "Memory manager não inicializado"}

    try:
        # Buscar análises anteriores relacionadas
        search_result = search_analysis_memory(
            query=current_question,
            dataset_name=dataset_name,
            limit=max_context_memories,
            namespace=namespace
        )

        if not search_result.get("success"):
            return search_result

        previous_analyses = search_result.get("results", [])

        # Buscar análises do mesmo dataset
        dataset_analyses = memory_manager.get_memories_by_tags(
            tags=[dataset_name],
            namespace=namespace
        )[:max_context_memories]

        # Formatear contexto
        context = {
            "current_question": current_question,
            "dataset_name": dataset_name,
            "related_analyses": previous_analyses,
            "dataset_history": [
                {
                    "analysis_type": mem.metadata.get("analysis_type"),
                    "question": mem.metadata.get("question"),
                    "key_insights": mem.metadata.get("insights", [])[:3],
                    "timestamp": mem.timestamp.isoformat()
                }
                for mem in dataset_analyses
            ],
            "context_summary": generate_context_summary(previous_analyses, dataset_analyses)
        }

        return {
            "success": True,
            "context": context,
            "has_previous_context": len(previous_analyses) > 0 or len(dataset_analyses) > 0
        }

    except Exception as e:
        logger.error(f"Erro ao recuperar contexto: {e}")
        return {"error": str(e)}


def generate_context_summary(related_analyses: List[Dict], dataset_analyses: List[MemoryEntry]) -> str:
    """Gera resumo do contexto disponível"""
    summary_parts = []

    if related_analyses:
        analysis_types = [a.get("analysis_type") for a in related_analyses]
        summary_parts.append(f"Análises similares encontradas: {', '.join(set(analysis_types))}")

    if dataset_analyses:
        total_analyses = len(dataset_analyses)
        summary_parts.append(f"Dataset já analisado {total_analyses} vez(es) anteriormente")

    return " | ".join(summary_parts) if summary_parts else "Nenhum contexto anterior encontrado"


# SQL Scripts para configuração do Supabase (executar manualmente no SQL Editor)

SUPABASE_SETUP_SCRIPT = """
-- Script de configuração do Supabase para memória semântica
-- Execute este script no SQL Editor do Supabase

-- 1. Habilitar extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Criar tabela para memórias dos agentes
CREATE TABLE IF NOT EXISTS agent_memories (
    id text PRIMARY KEY,
    content text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    embedding vector(1536),
    timestamp timestamptz DEFAULT timezone('utc'::text, now()) NOT NULL,
    tags text[] DEFAULT '{}'::text[],
    namespace text DEFAULT 'default'::text
);

-- 3. Criar índices para performance
CREATE INDEX IF NOT EXISTS agent_memories_embedding_idx ON agent_memories 
USING hnsw (embedding vector_ip_ops);

CREATE INDEX IF NOT EXISTS agent_memories_namespace_idx ON agent_memories (namespace);
CREATE INDEX IF NOT EXISTS agent_memories_tags_idx ON agent_memories USING gin (tags);
CREATE INDEX IF NOT EXISTS agent_memories_timestamp_idx ON agent_memories (timestamp);

-- 4. Criar função de busca por similaridade
CREATE OR REPLACE FUNCTION match_memories(
  query_embedding vector(1536),
  match_namespace text,
  match_count int DEFAULT 10,
  similarity_threshold float DEFAULT 0.7
)
RETURNS TABLE (
  id text,
  content text,
  metadata jsonb,
  embedding vector(1536),
  timestamp timestamptz,
  tags text[],
  namespace text,
  similarity float
)
LANGUAGE SQL STABLE
AS $$
  SELECT
    agent_memories.id,
    agent_memories.content,
    agent_memories.metadata,
    agent_memories.embedding,
    agent_memories.timestamp,
    agent_memories.tags,
    agent_memories.namespace,
    (agent_memories.embedding <#> query_embedding) * -1 as similarity
  FROM agent_memories
  WHERE 
    agent_memories.namespace = match_namespace
    AND (agent_memories.embedding <#> query_embedding) * -1 > similarity_threshold
  ORDER BY agent_memories.embedding <#> query_embedding
  LIMIT match_count;
$$;

-- 5. Configurar Row Level Security (RLS)
ALTER TABLE agent_memories ENABLE ROW LEVEL SECURITY;

-- 6. Criar políticas de acesso (ajustar conforme necessário)
CREATE POLICY "Permitir todas as operações para usuários autenticados" 
ON agent_memories 
FOR ALL 
TO authenticated 
USING (true);
"""
