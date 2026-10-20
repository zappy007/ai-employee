## 🏗️ System Architecture

```mermaid
graph TD
    User((Telegram User)) -->|Text/Voice/Files| Bot[Telegram Bot API]
    Bot --> Dispatcher{Rule-Based Dispatcher}
    
    Dispatcher -->|Fallback| Core[Core Chat + Memory]
    Dispatcher -->|Research| Res[Researcher Agent]
    Dispatcher -->|Planning| Plan[Planner Agent]
    Dispatcher -->|Drafting| Write[Writer Agent]
    Dispatcher -->|Calendar/Email| Exec[Executive Agent]
    
    Res -->|Web| Tavily[Tavily API]
    Res -->|LLM| Groq[Groq API]
    
    Plan -->|Memory| Mem0[mem0 / Qdrant]
    Plan -->|LLM| Groq
    
    Write -->|Memory| Mem0
    Write -->|LLM| Groq
    
    Exec -->|Calendar/Gmail| Google[Google Workspace API]
    Exec -->|LLM| Groq
    
    Core -->|LLM| Groq
    Core -->|Memory| Mem0
    
    Mem0 -->|Vector DB| Qdrant[(Qdrant Cloud)]
    
    subgraph Background Services
        Scheduler[Scheduler Loop] -->|Reports| Bot
        Reminders[Reminders Loop] -->|Alerts| Bot
    end
