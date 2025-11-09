# Web Implementation Guide

**Step-by-step guide to deploy the Research Semantic POC as a web application**

---

## 🚀 Quick Start (5 minutes)

```bash
# Clone repository
git clone https://github.com/ebaenamar/research-semantic-poc.git
cd research-semantic-poc

# Start with Docker Compose
docker-compose up -d

# Access application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
# MinIO Console: http://localhost:9001
```

---

## 📦 Project Structure

```
research-semantic-poc/
├── frontend/                 # React application
│   ├── src/
│   │   ├── components/      # UI components
│   │   ├── pages/           # Page components
│   │   ├── api/             # API client
│   │   └── hooks/           # Custom hooks
│   ├── package.json
│   └── Dockerfile
│
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── api/             # API routes
│   │   ├── core/            # Core logic (existing code)
│   │   ├── models/          # Database models
│   │   ├── schemas/         # Pydantic schemas
│   │   └── worker/          # Celery tasks
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml        # Docker orchestration
└── nginx.conf                # Reverse proxy config
```

---

## 🔧 Backend Implementation

### 1. FastAPI Application Setup

**File**: `backend/app/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Research Semantic POC API",
    version="1.0.0",
    description="API for hypothesis generation and validation"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from app.api import pipelines, hypotheses, feedback, datasets
app.include_router(pipelines.router, prefix="/api/v1/pipelines", tags=["pipelines"])
app.include_router(hypotheses.router, prefix="/api/v1/hypotheses", tags=["hypotheses"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["feedback"])
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])

@app.get("/")
def root():
    return {"message": "Research Semantic POC API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}
```

### 2. Pipeline API

**File**: `backend/app/api/pipelines.py`

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import uuid
from datetime import datetime

from app.schemas import PipelineCreate, PipelineResponse, PipelineStatus
from app.core.pipeline_executor import PipelineExecutor
from app.models import Pipeline
from app.database import get_db

router = APIRouter()

@router.post("/", response_model=PipelineResponse)
async def create_pipeline(pipeline: PipelineCreate, db=Depends(get_db)):
    """Create a new pipeline configuration"""
    
    pipeline_id = str(uuid.uuid4())
    
    # Save to database
    db_pipeline = Pipeline(
        id=pipeline_id,
        name=pipeline.name,
        description=pipeline.description,
        config=pipeline.config.dict(),
        status="created",
        created_at=datetime.utcnow()
    )
    db.add(db_pipeline)
    db.commit()
    
    return PipelineResponse(
        id=pipeline_id,
        name=pipeline.name,
        status="created",
        created_at=db_pipeline.created_at
    )

@router.post("/{pipeline_id}/start")
async def start_pipeline(
    pipeline_id: str,
    background_tasks: BackgroundTasks,
    db=Depends(get_db)
):
    """Start pipeline execution"""
    
    pipeline = db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    if pipeline.status == "running":
        raise HTTPException(status_code=400, detail="Pipeline already running")
    
    # Update status
    pipeline.status = "running"
    pipeline.started_at = datetime.utcnow()
    db.commit()
    
    # Execute in background
    background_tasks.add_task(execute_pipeline, pipeline_id, pipeline.config)
    
    return {"message": "Pipeline started", "pipeline_id": pipeline_id}

@router.get("/{pipeline_id}/status")
async def get_pipeline_status(pipeline_id: str, db=Depends(get_db)):
    """Get current pipeline status"""
    
    pipeline = db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return PipelineStatus(
        id=pipeline.id,
        status=pipeline.status,
        progress=pipeline.progress or 0,
        current_stage=pipeline.current_stage,
        started_at=pipeline.started_at,
        completed_at=pipeline.completed_at
    )

@router.get("/", response_model=List[PipelineResponse])
async def list_pipelines(db=Depends(get_db)):
    """List all pipelines"""
    
    pipelines = db.query(Pipeline).order_by(Pipeline.created_at.desc()).all()
    return pipelines
```

### 3. WebSocket for Real-Time Updates

**File**: `backend/app/api/websockets.py`

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, pipeline_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[pipeline_id] = websocket
    
    def disconnect(self, pipeline_id: str):
        if pipeline_id in self.active_connections:
            del self.active_connections[pipeline_id]
    
    async def send_update(self, pipeline_id: str, message: dict):
        if pipeline_id in self.active_connections:
            await self.active_connections[pipeline_id].send_json(message)

manager = ConnectionManager()

@router.websocket("/{pipeline_id}/stream")
async def pipeline_stream(websocket: WebSocket, pipeline_id: str):
    await manager.connect(pipeline_id, websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(pipeline_id)

# Usage in pipeline executor:
async def update_progress(pipeline_id: str, stage: str, progress: float):
    await manager.send_update(pipeline_id, {
        "type": "progress",
        "stage": stage,
        "progress": progress,
        "timestamp": datetime.utcnow().isoformat()
    })
```

### 4. Pipeline Executor (Integrates Existing Code)

**File**: `backend/app/core/pipeline_executor.py`

```python
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import existing code
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from embeddings import PaperEmbedder
from clustering import SemanticClusterer
from extraction import ClassificationValidator
from extraction.custom_criteria import CustomCriteriaValidator

async def execute_pipeline(pipeline_id: str, config: dict):
    """Execute the full pipeline"""
    
    try:
        # Stage 1: Load Data
        await update_progress(pipeline_id, "loading", 0.1)
        df = load_data(config['data_source'])
        
        # Stage 2: Generate Embeddings
        await update_progress(pipeline_id, "embedding", 0.3)
        embedder = PaperEmbedder(model_name=config['embedding_model'])
        embeddings = embedder.embed_papers(df)
        
        # Stage 3: Clustering
        await update_progress(pipeline_id, "clustering", 0.5)
        clusterer = SemanticClusterer(method=config['clustering_method'])
        
        # Filter NaN
        valid_mask = ~np.isnan(embeddings).any(axis=1)
        valid_embeddings = embeddings[valid_mask]
        valid_df = df[valid_mask].reset_index(drop=True)
        
        # Reduce and cluster
        reduced = clusterer.reduce_dimensions(
            valid_embeddings,
            n_components=config.get('umap_components', 10)
        )
        labels = clusterer.cluster_hdbscan(
            reduced,
            min_cluster_size=config.get('min_cluster_size', 15)
        )
        
        # Stage 4: Validation
        await update_progress(pipeline_id, "validation", 0.7)
        validator = ClassificationValidator()
        validation_results = validator.validate_all_clusters(valid_df, labels)
        
        # Custom criteria if configured
        if config.get('custom_criteria'):
            custom_validator = CustomCriteriaValidator()
            # Add criteria based on config
            custom_results = custom_validator.evaluate_all_clusters(valid_df, labels)
        
        # Stage 5: Generate Hypotheses
        await update_progress(pipeline_id, "hypotheses", 0.9)
        hypotheses = generate_hypotheses(valid_df, labels, validation_results, config)
        
        # Save results
        save_results(pipeline_id, {
            'clusters': labels.tolist(),
            'validation': validation_results,
            'hypotheses': hypotheses
        })
        
        # Complete
        await update_progress(pipeline_id, "complete", 1.0)
        update_pipeline_status(pipeline_id, "completed")
        
    except Exception as e:
        await update_progress(pipeline_id, "error", 0)
        update_pipeline_status(pipeline_id, "failed", error=str(e))
        raise
```

---

## 🎨 Frontend Implementation

### 1. React Setup

**File**: `frontend/src/App.tsx`

```typescript
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';

import Dashboard from './pages/Dashboard';
import PipelineConfig from './pages/PipelineConfig';
import PipelineMonitor from './pages/PipelineMonitor';
import Results from './pages/Results';
import HypothesisDetail from './pages/HypothesisDetail';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/pipeline/new" element={<PipelineConfig />} />
          <Route path="/pipeline/:id/monitor" element={<PipelineMonitor />} />
          <Route path="/results/:id" element={<Results />} />
          <Route path="/hypothesis/:id" element={<HypothesisDetail />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
```

### 2. API Client

**File**: `frontend/src/api/client.ts`

```typescript
const API_BASE = 'http://localhost:8000/api/v1';

export const api = {
  // Pipelines
  createPipeline: async (config: PipelineConfig) => {
    const response = await fetch(`${API_BASE}/pipelines`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return response.json();
  },
  
  startPipeline: async (pipelineId: string) => {
    const response = await fetch(`${API_BASE}/pipelines/${pipelineId}/start`, {
      method: 'POST'
    });
    return response.json();
  },
  
  getPipelineStatus: async (pipelineId: string) => {
    const response = await fetch(`${API_BASE}/pipelines/${pipelineId}/status`);
    return response.json();
  },
  
  // Results
  getResults: async (resultId: string) => {
    const response = await fetch(`${API_BASE}/results/${resultId}`);
    return response.json();
  },
  
  // Hypotheses
  getHypotheses: async () => {
    const response = await fetch(`${API_BASE}/hypotheses`);
    return response.json();
  }
};
```

### 3. Real-Time Monitor Component

**File**: `frontend/src/components/PipelineMonitor.tsx`

```typescript
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';

interface PipelineStatus {
  stage: string;
  progress: number;
  timestamp: string;
}

export const PipelineMonitor: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  
  useEffect(() => {
    // WebSocket connection
    const ws = new WebSocket(`ws://localhost:8000/api/v1/pipelines/${id}/stream`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStatus(data);
    };
    
    return () => ws.close();
  }, [id]);
  
  if (!status) return <div>Connecting...</div>;
  
  return (
    <div className="pipeline-monitor">
      <h2>Pipeline Progress</h2>
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${status.progress * 100}%` }}
        />
      </div>
      <p>Current Stage: {status.stage}</p>
      <p>Progress: {(status.progress * 100).toFixed(1)}%</p>
    </div>
  );
};
```

---

## 🐳 Docker Setup

### docker-compose.yml

```yaml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/research_poc
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
      - ./output:/app/output

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=research_poc
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

---

## 🚀 Deployment Steps

### 1. Local Development

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm start
```

### 2. Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f backend
```

### 3. Production Deployment

```bash
# Use production docker-compose
docker-compose -f docker-compose.prod.yml up -d

# With SSL (Let's Encrypt)
# Configure nginx.conf for HTTPS
# Use certbot for certificates
```

---

## 📊 Features Roadmap

### Phase 1 (MVP - 2 weeks)
- [x] Basic pipeline configuration
- [x] Pipeline execution
- [x] Results viewing
- [ ] User authentication
- [ ] Basic monitoring

### Phase 2 (Enhanced - 4 weeks)
- [ ] Real-time WebSocket updates
- [ ] Interactive visualizations
- [ ] Feedback loop interface
- [ ] Collaboration features
- [ ] Export functionality

### Phase 3 (Advanced - 8 weeks)
- [ ] AI-powered suggestions
- [ ] Multi-user collaboration
- [ ] Advanced analytics
- [ ] Mobile app (PWA)
- [ ] API marketplace

---

## 🔒 Security Considerations

- JWT authentication
- Rate limiting
- Input validation
- SQL injection prevention
- XSS protection
- CORS configuration
- HTTPS only in production

---

## 📈 Monitoring & Logging

- Application logs (structured JSON)
- Performance metrics (Prometheus)
- Error tracking (Sentry)
- User analytics (Plausible/Matomo)
- Database monitoring

---

**Document Version**: 1.0
**Repository**: https://github.com/ebaenamar/research-semantic-poc
**Support**: Open an issue on GitHub
