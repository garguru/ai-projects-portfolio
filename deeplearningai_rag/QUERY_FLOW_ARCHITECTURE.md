# RAG Query Processing Flow - Complete Architecture

## Visual Flow Diagram

```mermaid
graph TB
    subgraph "🌐 FRONTEND (Browser)"
        U[User Input] --> JS[script.js:sendMessage]
        JS --> F1[POST /api/query]
        F1 --> |JSON: query + session_id| API
    end

    subgraph "🚀 FASTAPI BACKEND"
        API[app.py:70<br/>@app.post /api/query] --> SM[Session Manager<br/>Create/Get Session]
        SM --> RAG[rag_system.py:122<br/>RAGSystem.query]
    end

    subgraph "🧠 RAG ORCHESTRATION"
        RAG --> H[Get Conversation<br/>History]
        RAG --> AI[ai_generator.py:70<br/>AIGenerator.generate_response]
        AI --> |Claude API| CL[Claude Sonnet 4<br/>with Tools]
    end

    subgraph "🔧 TOOL EXECUTION"
        CL --> |Tool Decision| TC{Tool Call?}
        TC -->|Yes| ST[search_tools.py:60<br/>CourseSearchTool]
        TC -->|No| DIRECT[Direct Response]
        ST --> TM[ToolManager<br/>Execute Search]
    end

    subgraph "💾 VECTOR SEARCH"
        TM --> VS[vector_store.py:280<br/>VectorStore.search]
        VS --> EMB[SentenceTransformers<br/>all-MiniLM-L6-v2]
        EMB --> |Embeddings| CH[(ChromaDB)]

        CH --> CC[course_content<br/>Collection]
        CH --> CAT[course_catalog<br/>Collection]

        CC --> |Top 5 Chunks| RES[Search Results<br/>+ Metadata]
    end

    subgraph "📝 RESPONSE GENERATION"
        RES --> |Context| CL2[Claude Final<br/>Response]
        DIRECT --> CL2
        CL2 --> RESP[Generated Answer<br/>+ Sources + Links]
    end

    subgraph "↩️ RESPONSE FLOW"
        RESP --> RAG2[Update Session<br/>History]
        RAG2 --> API2[Return QueryResponse]
        API2 --> |JSON Response| JS2[Frontend Receives]
        JS2 --> UI[Display in Chat<br/>with Sources]
    end

    style U fill:#e1f5fe
    style UI fill:#c8e6c9
    style CH fill:#fff3b2
    style CL fill:#ce93d8
    style CL2 fill:#ce93d8
```

## Detailed Component Breakdown

### 1️⃣ **User Interface Layer**
```
📍 Location: frontend/script.js
┌─────────────────────────────────┐
│  User types: "What is RAG?"     │
│  ↓                               │
│  sendMessage() → Line 61        │
│  ↓                               │
│  POST /api/query                │
│  Body: {query, session_id}      │
└─────────────────────────────────┘
```

### 2️⃣ **API Gateway**
```
📍 Location: backend/app.py
┌─────────────────────────────────┐
│  @app.post("/api/query")        │
│  Line 70-91                     │
│  ↓                               │
│  SessionManager handles ID      │
│  ↓                               │
│  rag_system.query()             │
└─────────────────────────────────┘
```

### 3️⃣ **RAG System Core**
```
📍 Location: backend/rag_system.py
┌─────────────────────────────────────┐
│  query() → Line 122-163             │
│  ├── Get conversation history       │
│  ├── Prepare prompt                 │
│  ├── Call AI with tools             │
│  └── Return (answer, sources, links)│
└─────────────────────────────────────┘
```

### 4️⃣ **AI Generation with Tools**
```
📍 Location: backend/ai_generator.py
┌─────────────────────────────────────┐
│  generate_response() → Line 50-120  │
│  ├── System prompt + history        │
│  ├── Tool definitions provided      │
│  ├── Claude decides: use tool?      │
│  │   ├── Yes: Execute up to 2x      │
│  │   └── No: Direct response        │
│  └── Final response generation      │
└─────────────────────────────────────┘
```

### 5️⃣ **Search Tool Execution**
```
📍 Location: backend/search_tools.py
┌─────────────────────────────────────┐
│  CourseSearchTool → Line 37-127     │
│  ├── Parse tool parameters          │
│  ├── Resolve course name            │
│  ├── Call vector_store.search()     │
│  └── Format results with sources    │
└─────────────────────────────────────┘
```

### 6️⃣ **Vector Database Layer**
```
📍 Location: backend/vector_store.py
┌─────────────────────────────────────┐
│  ChromaDB Collections:              │
│                                      │
│  📚 course_catalog                  │
│  ├── Course titles                  │
│  ├── Instructor names               │
│  └── Lesson metadata                │
│                                      │
│  📄 course_content                  │
│  ├── Text chunks (800 chars)        │
│  ├── Course/lesson attribution      │
│  └── Embeddings (384 dimensions)    │
└─────────────────────────────────────┘
```

## Data Flow Timeline

```
TIME    COMPONENT           ACTION
────────────────────────────────────────────────
0ms     Frontend           User submits query
10ms    FastAPI           Receives POST request
15ms    SessionManager    Creates/retrieves session
20ms    RAGSystem        Initiates query processing
25ms    AIGenerator      Calls Claude API
100ms   Claude           Analyzes query, decides to use tool
150ms   SearchTool       Executes search request
160ms   VectorStore      Embeds query text
170ms   ChromaDB         Performs similarity search
180ms   ChromaDB         Returns top 5 chunks
200ms   SearchTool       Formats results with sources
250ms   Claude           Generates response using context
800ms   RAGSystem        Updates session history
810ms   FastAPI          Returns JSON response
820ms   Frontend         Updates UI with answer
────────────────────────────────────────────────
```

## Key Features Illustrated

### 🎯 **Smart Routing**
- Claude decides whether to search or answer directly
- Tool-based architecture allows flexible workflows

### 🔍 **Semantic Search**
- Query → Embedding → Vector similarity
- Not keyword matching, but meaning matching

### 📚 **Dual Collection Strategy**
- `course_catalog`: Fast course resolution
- `course_content`: Detailed content search

### 🔄 **Session Management**
- Maintains conversation context
- Enables follow-up questions

### 📍 **Source Attribution**
- Every chunk tracked to course + lesson
- Direct links to DeepLearning.AI lessons

## Configuration
```python
# backend/config.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions
CHUNK_SIZE = 800                      # Characters per chunk
CHUNK_OVERLAP = 100                   # Overlap for context
MAX_RESULTS = 5                       # Top chunks returned
MAX_HISTORY = 2                       # Message pairs remembered
```

## Example Query Trace

**User Query**: "What is RAG?"

1. **Embedding**: [0.23, -0.45, 0.12, ...] (384 dims)
2. **ChromaDB Search**: Finds 5 similar chunks
3. **Retrieved Context**:
   - Chunk 1: "RAG stands for Retrieval Augmented Generation..."
   - Chunk 2: "The benefits of RAG include reducing hallucinations..."
4. **Claude Response**: Comprehensive explanation with sources
5. **Source Links**:
   - `courses/prompt-compression/lesson/1`
   - `courses/advanced-retrieval/lesson/0`

---

*This architecture enables intelligent, context-aware responses grounded in course materials while maintaining conversation flow and source transparency.*