# 🤖 CSV Agent System

Sistemax avançado de agentes de IA para análise conversacional de dados CSV usando CrewAI, Gemini Flash 2.5, e Supabase.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![CrewAI](https://img.shields.io/badge/CrewAI-0.80.0%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0%2B-red)
![Supabase](https://img.shields.io/badge/Supabase-Vector-orange)

## 🚀 Características Principais

- **💬 Análise Conversacional**: Faça perguntas naturais sobre seus dados CSV
- **🤖 Sistema Multi-Agentes**: 5 agentes especializados trabalhando em colaboração
- **🧠 Memória Semântica**: Contexto persistente usando Supabase Vector
- **📊 Visualizações Interativas**: Gráficos Plotly gerados automaticamente
- **⚡ Performance**: Processamento otimizado com UV e Gemini Flash 2.5
- **🎨 Interface Intuitiva**: Streamlit app moderno e responsivo

## 🏗️ Arquitetura do Sistema

### Agentes Especializados

1. **🎯 Orchestrator Agent**
   - Coordena a equipe de agentes
   - Interpreta perguntas do usuário
   - Planeja fluxos de execução
   - Consolida resultados finais

2. **📊 DataLoader Agent**
   - Ingestão e validação de CSV
   - Detecção automática de tipos
   - Relatórios de qualidade dos dados
   - Pré-processamento inteligente

3. **🔍 Analyzer Agent**
   - Análises estatísticas profundas
   - Detecção de outliers e padrões
   - Correlações e relacionamentos
   - Insights interpretativos

4. **📈 Visualization Agent**
   - Gráficos interativos Plotly
   - Dashboards dinâmicos
   - Visualizações contextuais
   - Explicações visuais

5. **🧠 Memory Agent**
   - Persistência com Supabase Vector
   - Busca semântica de contexto
   - Histórico de análises
   - Aprendizado contínuo

## 🛠️ Stack Tecnológico

- **Gerenciamento de Dependências**: UV (ultrafast Python package manager)
- **Framework de Agentes**: CrewAI 0.80+
- **LLM**: Google Gemini Flash 2.5
- **Interface**: Streamlit
- **Banco de Dados**: Supabase (PostgreSQL + pgvector)
- **Visualizações**: Plotly
- **Processamento**: Pandas, NumPy, Scikit-learn

## 📦 Instalação

### Pré-requisitos

- Python 3.9 a 3.12
- UV package manager
- Conta Google (para Gemini API)
- Conta Supabase

### 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/csv-agent-system.git
cd csv-agent-system
```

### 2. Instalar UV

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip
pip install uv
```

### 3. Configurar Ambiente

```bash
# Instalar dependências
uv sync

# Ativar ambiente virtual
source .venv/bin/activate  # Linux/macOS
# ou
.venv\Scripts\activate   # Windows
```

### 4. Configurar Variáveis de Ambiente

Copie `.env.example` para `.env` e configure:

```bash
cp .env.example .env
```

Edite o arquivo `.env`:

```env
# API Keys
GEMINI_API_KEY=sua_chave_gemini_aqui
OPENAI_API_KEY=sua_chave_openai_aqui

# Supabase Configuration
SUPABASE_URL=https://seu-projeto.supabase.co
SUPABASE_ANON_KEY=sua_chave_anonima
SUPABASE_SERVICE_ROLE_KEY=sua_chave_service_role

# Configurações adicionais...
```

### 5. Configurar Supabase

Execute no SQL Editor do Supabase:

```sql
-- Habilitar pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Criar tabela de memórias
CREATE TABLE IF NOT EXISTS agent_memories (
    id text PRIMARY KEY,
    content text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    embedding vector(1536),
    timestamp timestamptz DEFAULT timezone('utc'::text, now()) NOT NULL,
    tags text[] DEFAULT '{}'::text[],
    namespace text DEFAULT 'default'::text
);

-- Criar índices
CREATE INDEX IF NOT EXISTS agent_memories_embedding_idx ON agent_memories 
USING hnsw (embedding vector_ip_ops);

-- Função de busca por similaridade
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
```

## 🚦 Como Usar

### 1. Iniciar a Aplicação

```bash
uv run streamlit run csv_agent_system/ui/streamlit_app.py
```

### 2. Upload de Dados

- Faça upload de um arquivo CSV ou Excel
- Visualize o preview dos dados
- Confirme a estrutura detectada

### 3. Análise Conversacional

Exemplos de perguntas:

```
📊 Análise Descritiva:
- "Descreva estatisticamente meus dados"
- "Qual a distribuição da coluna vendas?"

🔗 Correlações:
- "Existe correlação entre preço e qualidade?"
- "Quais variáveis estão mais relacionadas?"

📈 Comparações:
- "Compare as vendas por categoria"
- "Qual região tem melhor performance?"

⏰ Tendências:
- "Analise as tendências ao longo do tempo"
- "Existem padrões sazonais?"

🎯 Outliers:
- "Identifique valores anômalos nos dados"
- "Quais são os outliers em vendas?"
```

### 4. Visualizações Interativas

- Gráficos são gerados automaticamente
- Hover para detalhes
- Zoom e pan disponíveis
- Exports em PNG/HTML

### 5. Memória de Contexto

- Análises são automaticamente salvas
- Contexto é recuperado em novas perguntas
- Histórico acessível na interface

## 🏗️ Estrutura do Projeto

```
csv_agent_system/
├── agents/                 # Agentes especializados
│   ├── orchestrator.py    # Coordenador principal
│   ├── data_loader.py     # Ingestão de dados
│   ├── analyzer.py        # Análises estatísticas
│   ├── visualization.py   # Criação de gráficos
│   └── memory.py          # Gestão de memória
├── config/                # Configurações
│   ├── settings.py        # Configurações globais
│   └── prompts.py         # Prompts dos agentes
├── tools/                 # Ferramentas especializadas
│   ├── csv_tools.py       # Processamento CSV
│   ├── plotting_tools.py  # Visualizações Plotly
│   └── memory_tools.py    # Ferramentas de memória
├── ui/                    # Interface Streamlit
│   └── streamlit_app.py   # Aplicação principal
├── utils/                 # Utilitários
│   ├── helpers.py         # Funções auxiliares
│   └── validators.py      # Validações
├── pyproject.toml         # Configuração UV/Python
├── .env.example           # Template de variáveis
└── README.md              # Este arquivo
```

## 🔧 Configuração Avançada

### Personalizando Agentes

```python
# config/prompts.py
CUSTOM_AGENT_PROMPT = {
    "role": "Seu papel personalizado",
    "goal": "Objetivo específico",
    "backstory": "Contexto e experiência",
    "instructions": "Instruções detalhadas"
}
```

### Adicionando Ferramentas

```python
# tools/custom_tools.py
from crewai_tools import tool

@tool("minha_ferramenta")
def minha_ferramenta_personalizada(dados):
    """Descrição da ferramenta"""
    # Implementação
    return resultado
```

### Configurações de Performance

```env
# .env
MAX_CSV_SIZE_MB=100
MAX_ANALYSIS_ITERATIONS=10
CREWAI_STORAGE_DIR=./custom_storage
MEMORY_NAMESPACE=meu_projeto
```

## 📊 Exemplos de Uso

### 1. Análise de Vendas

```python
# Upload: vendas_2024.csv
# Pergunta: "Analise as tendências de vendas por mês"
# Resultado: Série temporal + insights sazonais
```

### 2. Análise de Qualidade

```python
# Upload: produtos_feedback.csv  
# Pergunta: "Existe correlação entre preço e satisfação?"
# Resultado: Scatter plot + análise de correlação
```

### 3. Detecção de Anomalias

```python
# Upload: transacoes.csv
# Pergunta: "Identifique transações anômalas"
# Resultado: Boxplots + lista de outliers
```

## 🐛 Troubleshooting

### Problemas Comuns

**1. Erro de API Key**
```
AssertionError: GEMINI_API_KEY not found
```
- Verifique se as chaves estão configuradas no `.env`
- Confirme que o arquivo `.env` está no diretório raiz

**2. Erro de Conexão Supabase**
```
Could not connect to Supabase
```
- Verifique URL e chaves do Supabase
- Confirme que o projeto está ativo
- Execute o script SQL de configuração

**3. Erro de Encoding CSV**
```
UnicodeDecodeError
```
- Tente salvar o CSV em UTF-8
- Use a detecção automática de encoding
- Converta caracteres especiais

**4. Performance Lenta**
```
Análise demorada
```
- Reduza o tamanho do dataset
- Ajuste `MAX_ANALYSIS_ITERATIONS`
- Use amostragem para datasets grandes

### Logs e Debug

```bash
# Ativar logs detalhados
export CREWAI_LOG_LEVEL=DEBUG

# Verificar configurações
uv run python -c "from csv_agent_system.config.settings import settings; print(settings)"
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch: `git checkout -b feature/nova-feature`
3. Commit: `git commit -m 'Add nova feature'`
4. Push: `git push origin feature/nova-feature`
5. Abra um Pull Request

## 📄 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- [CrewAI](https://github.com/joaomdmoura/crewai) - Framework de agentes
- [Google](https://ai.google.dev/) - Gemini Flash 2.5 LLM
- [Supabase](https://supabase.com/) - Database e Vector Search
- [Streamlit](https://streamlit.io/) - Framework web
- [UV](https://github.com/astral-sh/uv) - Package manager

## 📞 Suporte

- 🐛 **Issues**: [GitHub Issues](https://github.com/seu-usuario/csv-agent-system/issues)
- 💬 **Discussões**: [GitHub Discussions](https://github.com/seu-usuario/csv-agent-system/discussions)
- 📧 **Email**: seu-email@exemplo.com

---

<div align="center">
  <strong>🚀 Feito com ❤️ usando CrewAI e tecnologias de ponta</strong>
</div>
