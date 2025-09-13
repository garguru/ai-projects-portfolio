# RAG Query Flow - Tree Structure

```
📁 RAG_QUERY_FLOW/
│
├── 🌐 FRONTEND/
│   ├── 📄 index.html
│   ├── 📄 script.js
│   │   ├── 🔵 sendMessage()                    Line: 61
│   │   ├── 🔵 POST /api/query                  Line: 79
│   │   └── 🔵 addMessage()                     Line: 101
│   │
│   └── 📊 REQUEST_PAYLOAD
│       ├── query: "What is RAG?"
│       └── session_id: "session_1"
│
├── 🚀 BACKEND/
│   ├── 📄 app.py
│   │   ├── 🟢 @app.post("/api/query")         Line: 70
│   │   ├── 🟢 query_documents()                Line: 70-91
│   │   └── 🟢 SessionManager.create()          Line: 76
│   │
│   ├── 📄 rag_system.py
│   │   ├── 🟣 RAGSystem.query()               Line: 122-163
│   │   ├── 🟣 get_conversation_history()       Line: 141
│   │   ├── 🟣 ai_generator.generate()          Line: 144
│   │   └── 🟣 session_manager.add_exchange()   Line: 160
│   │
│   ├── 📄 ai_generator.py
│   │   ├── 🔴 AIGenerator.generate_response()  Line: 50-120
│   │   ├── 🔴 client.messages.create()         Line: 99
│   │   ├── 🔴 Tool Decision Loop (2 rounds)    Line: 85-108
│   │   └── 🔴 handle_tool_execution()          Line: 103
│   │
│   ├── 📄 search_tools.py
│   │   ├── 🟠 CourseSearchTool.execute()      Line: 60-127
│   │   ├── 🟠 resolve_course_name()            Line: 78
│   │   ├── 🟠 vector_store.search()            Line: 87
│   │   └── 🟠 format_results()                 Line: 120
│   │
│   └── 📄 vector_store.py
│       ├── 🟡 VectorStore.search()            Line: 280-320
│       ├── 🟡 SentenceTransformers.encode()    Line: 290
│       └── 🟡 course_content.query()           Line: 305
│
├── 💾 CHROMADB/
│   ├── 📚 course_catalog/
│   │   ├── 📖 "Building Towards Computer Use"
│   │   ├── 📖 "MCP: Build Rich-Context AI Apps"
│   │   ├── 📖 "Advanced Retrieval for AI"
│   │   └── 📖 "Prompt Compression"
│   │
│   └── 📄 course_content/
│       ├── 🔍 Chunk[0]: "RAG stands for..."      800 chars
│       ├── 🔍 Chunk[1]: "benefits include..."    800 chars
│       ├── 🔍 Chunk[2]: "reducing hallucin..."   800 chars
│       └── 🔍 ... (528 total chunks)
│
├── 🤖 CLAUDE_API/
│   ├── 📡 REQUEST
│   │   ├── model: "claude-sonnet-4-20250514"
│   │   ├── messages: [{role: "user", content}]
│   │   ├── tools: [CourseSearchTool]
│   │   └── tool_choice: {type: "auto"}
│   │
│   └── 📡 RESPONSE
│       ├── stop_reason: "tool_use"
│       ├── tool_calls: [{name: "search_course"}]
│       └── final_content: "RAG stands for..."
│
└── 📊 DATA_FLOW/
    ├── ⏱️ TIMELINE
    │   ├── 0ms:    User submits query
    │   ├── 10ms:   FastAPI receives POST
    │   ├── 20ms:   RAGSystem initiates
    │   ├── 100ms:  Claude decides tool use
    │   ├── 160ms:  VectorStore embeds query
    │   ├── 180ms:  ChromaDB similarity search
    │   ├── 250ms:  Claude generates response
    │   └── 820ms:  Frontend displays result
    │
    └── 🔄 FLOW_SEQUENCE
        ├── 1️⃣ Frontend → API
        ├── 2️⃣ API → RAGSystem
        ├── 3️⃣ RAGSystem → AIGenerator
        ├── 4️⃣ AIGenerator → Claude
        ├── 5️⃣ Claude → SearchTool
        ├── 6️⃣ SearchTool → VectorStore
        ├── 7️⃣ VectorStore → ChromaDB
        ├── 8️⃣ ChromaDB → SearchResults
        ├── 9️⃣ SearchResults → Claude
        └── 🔟 Claude → Frontend

## Configuration Details

📁 backend/config.py
├── 🔧 EMBEDDING_MODEL: "all-MiniLM-L6-v2"     # 384 dimensions
├── 🔧 CHUNK_SIZE: 800                         # Characters
├── 🔧 CHUNK_OVERLAP: 100                      # Context preservation
├── 🔧 MAX_RESULTS: 5                          # Top chunks
├── 🔧 MAX_HISTORY: 2                          # Message pairs
└── 🔧 ANTHROPIC_MODEL: "claude-sonnet-4"      # AI model

## Session Storage

📁 SessionManager/
├── 📝 session_1/
│   ├── message_1: {user: "What is RAG?"}
│   ├── message_2: {assistant: "RAG stands..."}
│   └── timestamp: "2025-09-13T16:00:00Z"
│
└── 📝 session_2/
    └── ... (other sessions)

## Response Structure

📦 QueryResponse
├── 📜 answer: "RAG stands for Retrieval Augmented Generation..."
├── 📌 sources: [
│   ├── "Prompt Compression - Lesson 1"
│   ├── "Advanced Retrieval - Lesson 0"
│   └── "Prompt Compression - Lesson 1"
│   ]
├── 🔗 source_links: [
│   ├── "https://learn.deeplearning.ai/courses/prompt..."
│   └── "https://learn.deeplearning.ai/courses/advanced..."
│   ]
└── 🆔 session_id: "session_1"

## Key Components Status

✅ Frontend JavaScript     (Active)
✅ FastAPI Server         (Port 8000)
✅ RAG System            (Initialized)
✅ Claude API            (Connected)
✅ ChromaDB              (4 courses, 528 chunks)
✅ Search Tools          (2 tools registered)
✅ Session Manager       (Active sessions)

## Performance Metrics

📊 METRICS/
├── ⚡ Vector Search: ~20ms
├── ⚡ Embedding Generation: ~10ms
├── ⚡ Claude Response: ~650ms
├── ⚡ Total Response Time: ~820ms
└── ⚡ Memory Usage: ~250MB
```

---
*Tree structure showing complete RAG query flow from user input to response display*