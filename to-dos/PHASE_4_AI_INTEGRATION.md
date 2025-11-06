# PHASE 4: AI INTEGRATION

NLP pipeline, RAG system, knowledge graphs, and LLM-powered causal analysis.

**Timeline**: Q4 2025 (12 weeks, October-December 2025)
**Prerequisites**: Phase 1-3 complete, PostgreSQL, ML models deployed
**Team**: 2-3 developers (NLP specialist + AI engineer)
**Estimated Hours**: ~340 hours total

## Overview

| Sprint | Duration | Focus Area | Key Deliverables | Hours |
|--------|----------|------------|------------------|-------|
| Sprint 1 | Weeks 1-3 | NLP Pipeline | spaCy, SafeAeroBERT, NER | 90h |
| Sprint 2 | Weeks 4-6 | RAG System | FAISS, Claude/GPT-4 | 85h |
| Sprint 3 | Weeks 7-9 | Knowledge Graph | Neo4j, 50K+ entities | 85h |
| Sprint 4 | Weeks 10-12 | Causal Inference & Generation | DoWhy, LLM reports | 80h |

## Sprint 1: NLP Pipeline (Weeks 1-3, October 2025)

**Goal**: Process 10K+ accident narratives with 87-91% classification accuracy using fine-tuned BERT.

### Week 1: Text Preprocessing with spaCy

**Tasks**:
- [ ] Extract narratives from PostgreSQL (events.narrative, narratives table)
- [ ] Install spaCy with en_core_web_lg model
- [ ] Implement preprocessing pipeline: tokenization, lemmatization, POS tagging
- [ ] Remove stopwords and aviation-specific noise (regulatory jargon)
- [ ] Handle multi-line narratives and special characters
- [ ] Create clean text corpus (10K+ narratives)
- [ ] Generate text statistics: avg length, vocabulary size, common n-grams

**Deliverables**:
- Cleaned narrative corpus (10K+ documents)
- spaCy preprocessing pipeline (reusable)
- Text statistics report (word frequency, length distribution)

**Success Metrics**:
- Process 10K narratives in <5 minutes
- Vocabulary size: 15K-20K unique tokens
- Average narrative length: 200-400 words

**Code Example**:
```python
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

class NarrativePreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

        # Add aviation-specific stopwords
        self.aviation_stop = {'aircraft', 'flight', 'pilot', 'reported', 'stated'}
        self.stop_words = STOP_WORDS.union(self.aviation_stop)

    def preprocess(self, text):
        """Clean and normalize narrative text"""
        # Process with spaCy
        doc = self.nlp(text.lower())

        # Lemmatization, remove stopwords and punctuation
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.is_alpha
            and len(token.text) > 2
        ]

        return ' '.join(tokens)

    def extract_entities(self, text):
        """Extract named entities (locations, organizations, dates)"""
        doc = self.nlp(text)

        entities = {
            'locations': [ent.text for ent in doc.ents if ent.label_ == 'GPE'],
            'dates': [ent.text for ent in doc.ents if ent.label_ == 'DATE'],
            'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        }

        return entities

# Usage
preprocessor = NarrativePreprocessor()

narratives = pd.read_sql("SELECT narrative FROM events WHERE narrative IS NOT NULL", engine)

narratives['clean_text'] = narratives['narrative'].apply(preprocessor.preprocess)
narratives['entities'] = narratives['narrative'].apply(preprocessor.extract_entities)
```

**Dependencies**: spacy, pandas, sqlalchemy

### Week 2: SafeAeroBERT Fine-Tuning

**Tasks**:
- [ ] Load pre-trained BERT model (bert-base-uncased or aviation-specific if available)
- [ ] Create labeled dataset: classify narratives by severity (fatal/serious/minor/none)
- [ ] Split dataset: 80% train, 10% validation, 10% test
- [ ] Fine-tune BERT with Hugging Face Transformers
- [ ] Train for 3-5 epochs with early stopping
- [ ] Optimize hyperparameters: learning rate, batch size, warmup steps
- [ ] Evaluate on test set: accuracy, precision, recall, F1

**Deliverables**:
- Fine-tuned SafeAeroBERT model (87-91% accuracy)
- Model saved to Hugging Face Hub or local storage
- Classification report and confusion matrix

**Success Metrics**:
- Test accuracy: 87-91% (target based on aviation NLP research)
- F1-score: >0.85
- Training time: <2 hours on GPU

**Research Finding**: Aviation-specific BERT models are emerging in research. A 2024 study on AI in aviation safety showed transformer models achieving 80%+ accuracy on accident classification tasks. Fine-tuning on domain-specific data is critical for performance.

**Code Example**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load pre-trained BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Prepare dataset
from datasets import Dataset

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Create dataset
train_df = narratives[narratives['split'] == 'train']
train_dataset = Dataset.from_pandas(train_df[['clean_text', 'severity_label']])
train_dataset = train_dataset.rename_column('clean_text', 'text')
train_dataset = train_dataset.rename_column('severity_label', 'labels')
train_dataset = train_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir='./logs',
    save_strategy='epoch',
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate(test_dataset)
print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
print(f"Test F1: {results['eval_f1']:.4f}")

# Save model
model.save_pretrained('./models/safe_aero_bert')
tokenizer.save_pretrained('./models/safe_aero_bert')
```

**Dependencies**: transformers, datasets, torch, sklearn

### Week 3: Custom NER & Topic Modeling

**Tasks**:
- [ ] Train custom NER model for aviation entities: aircraft types, airports, weather, failures
- [ ] Annotate 1000+ narratives with aviation-specific entities (use Label Studio)
- [ ] Fine-tune spaCy NER on annotated data
- [ ] Implement BERTopic for unsupervised topic discovery
- [ ] Extract 20-30 topics from narratives (e.g., engine failure, weather, pilot error)
- [ ] Visualize topics with pyLDAvis or BERTopic visualizations
- [ ] Integrate NER and topics into PostgreSQL (new columns)

**Deliverables**:
- Custom aviation NER model (>80% F1)
- BERTopic model with 20-30 interpretable topics
- Database updated with NER entities and topic assignments

**Success Metrics**:
- NER F1-score: >0.80 on test set
- Topic coherence: >0.4 (C_v coherence score)
- 90%+ of narratives assigned to topics

**Code Example**:
```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# BERTopic for topic modeling
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=embedding_model, nr_topics=30)

# Fit on narratives
topics, probs = topic_model.fit_transform(narratives['clean_text'].tolist())

# Visualize
topic_model.visualize_topics()
topic_model.visualize_barchart(top_n_topics=10)

# Get topic keywords
for topic_id in range(30):
    words = topic_model.get_topic(topic_id)
    print(f"Topic {topic_id}: {words[:5]}")

# Custom NER training
import spacy
from spacy.training import Example

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Add labels
ner.add_label("AIRCRAFT_TYPE")
ner.add_label("AIRPORT")
ner.add_label("WEATHER")
ner.add_label("FAILURE_MODE")

# Training data (annotated)
TRAIN_DATA = [
    ("The Cessna 172 crashed at KJFK during thunderstorms due to engine failure.",
     {"entities": [(4, 15, "AIRCRAFT_TYPE"), (28, 32, "AIRPORT"),
                   (40, 53, "WEATHER"), (61, 75, "FAILURE_MODE")]}),
    # ... 1000+ examples
]

# Train
for epoch in range(30):
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], losses=losses)

    print(f"Epoch {epoch}: {losses}")

# Save model
nlp.to_disk("./models/aviation_ner")
```

**Dependencies**: bertopic, sentence-transformers, spacy, label-studio

**Sprint 1 Total Hours**: 90 hours

---

## Sprint 2: RAG System (Weeks 4-6, November 2025)

**Goal**: Build retrieval-augmented generation system for querying 10K+ accident reports.

### Week 4: Document Vectorization

**Tasks**:
- [ ] Select embedding model: sentence-transformers (all-MiniLM-L6-v2 or all-mpnet-base-v2)
- [ ] Generate embeddings for 10K+ narratives (384 or 768 dimensions)
- [ ] Chunk long narratives: 512 tokens per chunk with 50 token overlap
- [ ] Store embeddings in vector database: FAISS (local) or Chroma (persistent)
- [ ] Create metadata: ev_id, date, severity, aircraft_type, location
- [ ] Build FAISS index: IndexFlatL2 or IndexIVFFlat for faster search
- [ ] Test retrieval: query "engine failure" → return top 10 similar narratives

**Deliverables**:
- 10K+ narrative embeddings
- FAISS/Chroma vector database
- Retrieval API: POST /rag/search

**Success Metrics**:
- Embedding generation: <10 minutes for 10K narratives
- Retrieval latency: <100ms for top-k search
- Retrieval precision@5: >0.7 (manual validation on 100 queries)

**Research Finding**: RAG evaluation metrics (2024) - Key metrics include precision@k, recall@k, MRR (Mean Reciprocal Rank), and NDCG (Normalized Discounted Cumulative Gain). For production RAG, target precision@5 > 0.7 and recall@10 > 0.8.

**Code Example**:
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class NarrativeVectorDB:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.metadata = []

    def chunk_text(self, text, chunk_size=512, overlap=50):
        """Split long narratives into chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def build_index(self, narratives_df):
        """Generate embeddings and build FAISS index"""
        all_chunks = []
        all_metadata = []

        for idx, row in narratives_df.iterrows():
            chunks = self.chunk_text(row['narrative'])

            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'ev_id': row['ev_id'],
                    'date': row['event_date'],
                    'severity': row['severity'],
                    'text': chunk
                })

        # Generate embeddings
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)

        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        self.metadata = all_metadata

        print(f"Indexed {len(all_chunks)} chunks from {len(narratives_df)} narratives")

    def search(self, query, top_k=10):
        """Retrieve top-k similar narratives"""
        query_embedding = self.model.encode([query]).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)

        results = [
            {
                **self.metadata[idx],
                'similarity': 1 / (1 + distances[0][i])  # Convert distance to similarity
            }
            for i, idx in enumerate(indices[0])
        ]

        return results

# Usage
vector_db = NarrativeVectorDB()
vector_db.build_index(narratives)

# Test retrieval
results = vector_db.search("engine failure during takeoff", top_k=5)
for r in results:
    print(f"[{r['ev_id']}] {r['date']}: {r['text'][:200]}... (score: {r['similarity']:.3f})")
```

**Dependencies**: sentence-transformers, faiss-cpu, numpy, pandas

### Week 5: LLM Integration (Claude/GPT-4)

**Tasks**:
- [ ] Set up Anthropic Claude API (or OpenAI GPT-4)
- [ ] Implement RAG pipeline: retrieve → augment → generate
- [ ] Design prompts: "Based on these accident reports: {context}, answer: {query}"
- [ ] Add citation: include ev_id references in LLM responses
- [ ] Implement streaming responses for real-time UX
- [ ] Add safety guardrails: filter sensitive content, validate responses
- [ ] Test on 50+ queries (manual evaluation)

**Deliverables**:
- RAG API endpoint: POST /rag/query
- Prompt templates for accident analysis
- Response validation & citation system

**Success Metrics**:
- Response relevance: >80% (manual evaluation)
- Citation accuracy: 100% (all references valid)
- Response time: <3 seconds (including retrieval + LLM)

**Code Example**:
```python
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

class RAGSystem:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def query(self, user_query, top_k=5):
        """RAG pipeline: retrieve → augment → generate"""
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_db.search(user_query, top_k=top_k)

        # Step 2: Format context
        context = "\n\n".join([
            f"[Accident {doc['ev_id']}, {doc['date']}]:\n{doc['text']}"
            for doc in retrieved_docs
        ])

        # Step 3: Create prompt
        prompt = f"""You are an aviation safety analyst. Based on the following NTSB accident reports, provide a detailed analysis.

Retrieved Reports:
{context}

User Question: {user_query}

Instructions:
1. Analyze the patterns across the retrieved reports
2. Cite specific accident IDs in your response
3. Provide actionable safety recommendations
4. If the question cannot be answered from the reports, clearly state that

Analysis:"""

        # Step 4: Generate response with Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20250929",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        # Step 5: Extract citations
        citations = [doc['ev_id'] for doc in retrieved_docs]

        return {
            "answer": response.content[0].text,
            "citations": citations,
            "retrieved_documents": retrieved_docs
        }

# FastAPI endpoint
from fastapi import APIRouter

router = APIRouter()
rag_system = RAGSystem(vector_db)

@router.post("/rag/query")
async def rag_query(query: str, top_k: int = 5):
    result = rag_system.query(query, top_k=top_k)

    return {
        "query": query,
        "answer": result['answer'],
        "citations": result['citations'],
        "sources": [
            {"ev_id": doc['ev_id'], "date": doc['date'], "snippet": doc['text'][:200]}
            for doc in result['retrieved_documents']
        ]
    }
```

**Dependencies**: anthropic, fastapi, sentence-transformers

### Week 6: RAG Evaluation & Optimization

**Tasks**:
- [ ] Create evaluation dataset: 100 query-answer pairs (gold standard)
- [ ] Implement RAG metrics: precision@k, recall@k, MRR, NDCG
- [ ] Evaluate retrieval: calculate precision@5, recall@10
- [ ] Evaluate generation: BLEU, ROUGE, BERTScore (compare to gold answers)
- [ ] Experiment with hybrid search: BM25 + vector search
- [ ] Implement re-ranking: cross-encoder to improve top-k results
- [ ] Optimize chunk size and overlap (ablation study)

**Deliverables**:
- RAG evaluation report (retrieval + generation metrics)
- Optimized RAG system (hybrid search + re-ranking)
- Benchmark results dashboard

**Success Metrics**:
- Retrieval precision@5: >0.70
- Retrieval recall@10: >0.80
- Generation ROUGE-L: >0.50
- End-to-end answer quality: >75% (human evaluation)

**Research Finding**: RAG evaluation best practices (2024) - Use precision@k and recall@k for retrieval, ROUGE/BLEU for generation. Hybrid search (BM25 + dense vectors) often outperforms vector search alone. Re-ranking with cross-encoders improves top-5 results by 10-20%.

**Code Example**:
```python
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class HybridRAGSystem:
    def __init__(self, vector_db):
        self.vector_db = vector_db

        # BM25 for lexical search
        corpus = [doc['text'] for doc in vector_db.metadata]
        tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Cross-encoder for re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def hybrid_search(self, query, top_k=10):
        """Combine BM25 and vector search"""
        # BM25 search
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_k = np.argsort(bm25_scores)[-top_k:]

        # Vector search
        vector_results = self.vector_db.search(query, top_k=top_k)
        vector_indices = [r['idx'] for r in vector_results]

        # Combine (simple union, could use weighted fusion)
        combined_indices = list(set(bm25_top_k).union(set(vector_indices)))

        # Retrieve documents
        candidates = [self.vector_db.metadata[idx] for idx in combined_indices]

        # Re-rank with cross-encoder
        pairs = [[query, doc['text']] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Sort by rerank score
        ranked_docs = sorted(zip(candidates, rerank_scores),
                             key=lambda x: x[1], reverse=True)

        return [doc for doc, score in ranked_docs[:top_k]]

# Evaluation metrics
def evaluate_retrieval(gold_data, rag_system, k=5):
    """Calculate precision@k and recall@k"""
    precisions = []
    recalls = []

    for item in gold_data:
        query = item['query']
        relevant_docs = set(item['relevant_ev_ids'])

        retrieved_docs = rag_system.hybrid_search(query, top_k=k)
        retrieved_ev_ids = set([doc['ev_id'] for doc in retrieved_docs])

        tp = len(retrieved_ev_ids.intersection(relevant_docs))
        precision = tp / k if k > 0 else 0
        recall = tp / len(relevant_docs) if relevant_docs else 0

        precisions.append(precision)
        recalls.append(recall)

    return {
        f'precision@{k}': np.mean(precisions),
        f'recall@{k}': np.mean(recalls)
    }
```

**Dependencies**: rank-bm25, sentence-transformers, numpy

**Sprint 2 Total Hours**: 85 hours

---

## Sprint 3: Knowledge Graph (Weeks 7-9, November-December 2025)

**Goal**: Build Neo4j knowledge graph with 50K+ entities and 100K+ relationships.

### Week 7: Neo4j Setup & Schema Design

**Tasks**:
- [ ] Install Neo4j Community Edition or Neo4j AuraDB (cloud)
- [ ] Design graph schema: node types (Accident, Aircraft, Airport, Person, Cause, Weather)
- [ ] Define relationships: INVOLVED_IN, OCCURRED_AT, CAUSED_BY, CONTRIBUTED_TO
- [ ] Create constraints and indexes for performance
- [ ] Load initial data: events, aircraft, airports from PostgreSQL
- [ ] Create Cypher queries for common patterns
- [ ] Test graph traversals: find all accidents at airport X

**Deliverables**:
- Neo4j database with schema
- Initial data load (10K+ accidents as nodes)
- Cypher query library (20+ common queries)

**Success Metrics**:
- Database size: 50K+ nodes, 100K+ relationships
- Query performance: <100ms for single-hop traversals
- Graph connectivity: 80%+ of nodes connected

**Research Finding**: Knowledge graph best practices (2024) - Use property graph model (Neo4j) for flexibility. Assign unique IDs to each node and reuse them for relationships. Use indexes on frequently queried properties (ev_id, airport_code, date).

**Code Example**:
```python
from neo4j import GraphDatabase

class AviationKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_schema(self):
        """Create constraints and indexes"""
        with self.driver.session() as session:
            # Constraints (unique IDs)
            session.run("CREATE CONSTRAINT accident_id IF NOT EXISTS ON (a:Accident) ASSERT a.ev_id IS UNIQUE")
            session.run("CREATE CONSTRAINT aircraft_id IF NOT EXISTS ON (ac:Aircraft) ASSERT ac.aircraft_key IS UNIQUE")
            session.run("CREATE CONSTRAINT airport_id IF NOT EXISTS ON (ap:Airport) ASSERT ap.airport_code IS UNIQUE")

            # Indexes
            session.run("CREATE INDEX accident_date IF NOT EXISTS FOR (a:Accident) ON (a.event_date)")
            session.run("CREATE INDEX aircraft_type IF NOT EXISTS FOR (ac:Aircraft) ON (ac.aircraft_category)")

    def load_accidents(self, accidents_df):
        """Load accidents as nodes"""
        with self.driver.session() as session:
            for _, row in accidents_df.iterrows():
                session.run(
                    """
                    MERGE (a:Accident {ev_id: $ev_id})
                    SET a.date = date($date),
                        a.latitude = $lat,
                        a.longitude = $lon,
                        a.severity = $severity,
                        a.narrative = $narrative
                    """,
                    ev_id=row['ev_id'],
                    date=row['event_date'].isoformat(),
                    lat=row['latitude'],
                    lon=row['longitude'],
                    severity=row['severity'],
                    narrative=row['narrative']
                )

    def create_relationships(self, aircraft_df):
        """Create INVOLVED_IN relationships"""
        with self.driver.session() as session:
            for _, row in aircraft_df.iterrows():
                session.run(
                    """
                    MATCH (a:Accident {ev_id: $ev_id})
                    MERGE (ac:Aircraft {aircraft_key: $aircraft_key})
                    SET ac.make = $make,
                        ac.model = $model,
                        ac.year = $year
                    MERGE (ac)-[:INVOLVED_IN]->(a)
                    """,
                    ev_id=row['ev_id'],
                    aircraft_key=row['aircraft_key'],
                    make=row['make'],
                    model=row['model'],
                    year=row['year_manufactured']
                )

    def query_accidents_at_airport(self, airport_code):
        """Find all accidents at a specific airport"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (ap:Airport {code: $code})<-[:OCCURRED_AT]-(a:Accident)
                RETURN a.ev_id, a.date, a.severity
                ORDER BY a.date DESC
                LIMIT 50
                """,
                code=airport_code
            )
            return [dict(record) for record in result]
```

**Dependencies**: neo4j, pandas

### Week 8: Entity & Relationship Extraction

**Tasks**:
- [ ] Extract entities from narratives using custom NER (from Sprint 1)
- [ ] Identify relationships using dependency parsing (spaCy)
- [ ] Implement pattern matching: "aircraft X experienced Y" → (Aircraft)-[:EXPERIENCED]->(Failure)
- [ ] Use LLM for relationship extraction: prompt Claude to identify cause-effect
- [ ] Create entity resolution: merge duplicate entities (e.g., "Cessna 172" vs "C172")
- [ ] Load entities and relationships into Neo4j (bulk import)
- [ ] Validate graph: check for disconnected components

**Deliverables**:
- Entity extraction pipeline (95%+ recall for key entities)
- Relationship extraction pipeline (80%+ precision)
- Populated knowledge graph (50K+ entities, 100K+ relationships)

**Success Metrics**:
- Entity extraction recall: >0.95 (find 95%+ of aircraft, airports, causes)
- Relationship extraction precision: >0.80 (80%+ of relationships are correct)
- Graph density: average degree > 2 (each node has 2+ connections)

**Code Example**:
```python
import spacy
from anthropic import Anthropic

class EntityRelationshipExtractor:
    def __init__(self, ner_model_path):
        self.nlp = spacy.load(ner_model_path)
        self.client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

    def extract_entities(self, narrative):
        """Extract aviation entities"""
        doc = self.nlp(narrative)

        entities = {
            'aircraft': [],
            'airports': [],
            'failures': [],
            'weather': [],
            'persons': []
        }

        for ent in doc.ents:
            if ent.label_ == 'AIRCRAFT_TYPE':
                entities['aircraft'].append(ent.text)
            elif ent.label_ == 'AIRPORT':
                entities['airports'].append(ent.text)
            elif ent.label_ == 'FAILURE_MODE':
                entities['failures'].append(ent.text)
            elif ent.label_ == 'WEATHER':
                entities['weather'].append(ent.text)
            elif ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)

        return entities

    def extract_relationships_llm(self, narrative):
        """Use LLM to extract cause-effect relationships"""
        prompt = f"""Analyze this aviation accident narrative and extract cause-effect relationships.

Narrative: {narrative}

Extract relationships in this JSON format:
[
    {{"subject": "entity1", "predicate": "CAUSED_BY", "object": "entity2"}},
    {{"subject": "entity3", "predicate": "CONTRIBUTED_TO", "object": "entity4"}}
]

Only extract relationships explicitly stated in the narrative. Common predicates:
- CAUSED_BY (direct cause)
- CONTRIBUTED_TO (contributing factor)
- OCCURRED_DURING (temporal)
- EXPERIENCED (aircraft experiencing a condition)

JSON relationships:"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20250929",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON response
        import json
        relationships = json.loads(response.content[0].text)

        return relationships

    def load_to_neo4j(self, kg, ev_id, entities, relationships):
        """Load entities and relationships to Neo4j"""
        with kg.driver.session() as session:
            # Create entity nodes
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    session.run(
                        f"MERGE (e:{entity_type.capitalize()} {{name: $name}})",
                        name=entity
                    )

            # Create relationships
            for rel in relationships:
                session.run(
                    f"""
                    MATCH (s {{name: $subject}})
                    MATCH (o {{name: $object}})
                    MERGE (s)-[:{rel['predicate']}]->(o)
                    """,
                    subject=rel['subject'],
                    object=rel['object']
                )
```

**Dependencies**: spacy, anthropic, neo4j

### Week 9: Graph Algorithms & Querying

**Tasks**:
- [ ] Implement PageRank: identify most influential entities (airports, aircraft types, causes)
- [ ] Community detection: find clusters of related accidents (Louvain algorithm)
- [ ] Shortest path: find chains of causation (A caused B, B caused C, ...)
- [ ] Centrality metrics: betweenness, closeness, degree
- [ ] Create graph analytics dashboard (Neo4j Bloom or custom)
- [ ] API endpoints: GET /kg/entity/{id}, GET /kg/relationships, POST /kg/query
- [ ] Generate graph visualizations (networkx, plotly)

**Deliverables**:
- Graph algorithms results (PageRank, communities, centrality)
- Knowledge graph API (5+ endpoints)
- Graph visualization dashboard

**Success Metrics**:
- PageRank converges within 20 iterations
- Detect 10-20 meaningful communities
- API latency: <200ms for single-entity queries

**Code Example**:
```python
def run_graph_algorithms(kg):
    """Run graph algorithms in Neo4j"""
    with kg.driver.session() as session:
        # PageRank (identify influential nodes)
        session.run(
            """
            CALL gds.pageRank.write({
                nodeProjection: '*',
                relationshipProjection: '*',
                writeProperty: 'pagerank'
            })
            """
        )

        # Get top 20 by PageRank
        result = session.run(
            """
            MATCH (n)
            RETURN n.name AS name, n.pagerank AS score, labels(n) AS type
            ORDER BY score DESC
            LIMIT 20
            """
        )

        print("Top 20 influential entities:")
        for record in result:
            print(f"{record['name']} ({record['type']}): {record['score']:.4f}")

        # Community detection (Louvain)
        session.run(
            """
            CALL gds.louvain.write({
                nodeProjection: '*',
                relationshipProjection: '*',
                writeProperty: 'community'
            })
            """
        )

        # Count communities
        result = session.run("MATCH (n) RETURN DISTINCT n.community AS community, count(*) AS size")

        print("\nCommunities detected:")
        for record in result:
            print(f"Community {record['community']}: {record['size']} nodes")

# FastAPI endpoints
@router.get("/kg/entity/{entity_name}")
async def get_entity(entity_name: str):
    """Get entity and its relationships"""
    with kg.driver.session() as session:
        result = session.run(
            """
            MATCH (e {name: $name})
            OPTIONAL MATCH (e)-[r]->(related)
            RETURN e, collect({type: type(r), node: related.name}) AS relationships
            """,
            name=entity_name
        )

        record = result.single()
        return {
            "entity": dict(record['e']),
            "relationships": record['relationships']
        }
```

**Dependencies**: neo4j, networkx (for visualization)

**Sprint 3 Total Hours**: 85 hours

---

## Sprint 4: Causal Inference & Generation (Weeks 10-12, December 2025)

**Goal**: Implement causal analysis with DoWhy and LLM-powered report generation.

### Week 10: Causal Inference with DoWhy

**Tasks**:
- [ ] Install DoWhy for causal inference
- [ ] Define causal graph: treatment (e.g., pilot experience), outcome (accident severity)
- [ ] Identify confounders: weather, aircraft age, airport complexity
- [ ] Estimate causal effects: average treatment effect (ATE), conditional ATE
- [ ] Validate assumptions: backdoor criterion, common cause
- [ ] Refute causal estimates: placebo test, random common cause
- [ ] Generate causal analysis report for top 10 factors

**Deliverables**:
- DoWhy causal models for 10+ treatment-outcome pairs
- Causal effect estimates with confidence intervals
- Refutation test results (validate causal claims)

**Success Metrics**:
- Identify 10+ statistically significant causal relationships (p < 0.05)
- Refutation tests pass (estimates stable under perturbations)
- Causal effects align with aviation safety literature

**Code Example**:
```python
import dowhy
from dowhy import CausalModel

# Define causal graph
causal_graph = """
digraph {
    pilot_experience -> accident_severity;
    weather -> accident_severity;
    aircraft_age -> accident_severity;
    weather -> pilot_experience;
    airport_complexity -> accident_severity;
}
"""

# Create causal model
model = CausalModel(
    data=df,
    treatment='pilot_experience',
    outcome='accident_severity',
    graph=causal_graph
)

# Identify causal effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate causal effect (backdoor adjustment)
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)

print(f"Causal Effect (ATE): {estimate.value:.4f}")
print(f"95% CI: [{estimate.get_confidence_intervals()[0]:.4f}, "
      f"{estimate.get_confidence_intervals()[1]:.4f}]")

# Refute estimates
refute_random_common_cause = model.refute_estimate(
    identified_estimand, estimate,
    method_name="random_common_cause"
)

refute_placebo = model.refute_estimate(
    identified_estimand, estimate,
    method_name="placebo_treatment_refuter"
)

print(f"\nRefutation (random common cause): {refute_random_common_cause}")
print(f"Refutation (placebo): {refute_placebo}")
```

**Dependencies**: dowhy, pandas, numpy

### Week 11: LLM-Powered Report Generation

**Tasks**:
- [ ] Design report template: Executive Summary, Findings, Causal Analysis, Recommendations
- [ ] Implement LLM-based report writer: aggregate accident data → generate narrative
- [ ] Use structured prompts: "Generate executive summary for 50 accidents in Q3 2024..."
- [ ] Include visualizations: embed matplotlib figures, SHAP plots, graphs
- [ ] Validate generated text: check for hallucinations, factual errors
- [ ] Create automated investigation summary for each accident
- [ ] Integrate with Phase 2 PDF report generator

**Deliverables**:
- LLM report generation pipeline
- Automated investigation summaries (1-2 pages per accident)
- Quality evaluation (BLEU, ROUGE scores vs human-written reports)

**Success Metrics**:
- BLEU score: >0.30 (compared to NTSB reports)
- ROUGE-L score: >0.40
- Human evaluation: 75%+ reports rated "good" or "excellent"

**Code Example**:
```python
class AccidentReportGenerator:
    def __init__(self, client):
        self.client = client

    def generate_executive_summary(self, accidents_df):
        """Generate executive summary from accident data"""
        # Aggregate statistics
        total_accidents = len(accidents_df)
        fatal_accidents = (accidents_df['severity'] == 'fatal').sum()
        top_causes = accidents_df['primary_cause'].value_counts().head(5).to_dict()

        prompt = f"""You are an aviation safety analyst writing an executive summary for a quarterly report.

Data Summary:
- Total accidents: {total_accidents}
- Fatal accidents: {fatal_accidents}
- Top 5 causes: {top_causes}

Write a 3-paragraph executive summary covering:
1. Overall trends (increase/decrease, severity distribution)
2. Key findings (top causes, notable patterns)
3. Priority recommendations

Executive Summary:"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20250929",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_accident_summary(self, ev_id):
        """Generate detailed summary for single accident"""
        # Fetch accident data
        accident = fetch_accident(ev_id)
        narrative = accident['narrative']
        findings = fetch_findings(ev_id)
        causal_factors = extract_causal_factors(ev_id)

        prompt = f"""Generate a concise accident investigation summary.

Event ID: {ev_id}
Date: {accident['event_date']}
Location: {accident['location']}
Aircraft: {accident['aircraft_make']} {accident['aircraft_model']}

Narrative:
{narrative}

Investigation Findings:
{findings}

Causal Factors:
{causal_factors}

Write a 2-paragraph summary:
1. What happened (sequence of events, injuries/damage)
2. Why it happened (probable cause, contributing factors)

Summary:"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20250929",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

# Evaluate quality
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def evaluate_generated_reports(generated, references):
    """Evaluate generated reports vs human-written"""
    bleu_scores = []
    rouge = Rouge()

    for gen, ref in zip(generated, references):
        # BLEU score
        bleu = sentence_bleu([ref.split()], gen.split())
        bleu_scores.append(bleu)

    # ROUGE scores
    rouge_scores = rouge.get_scores(generated, references, avg=True)

    return {
        'bleu': np.mean(bleu_scores),
        'rouge_l_f1': rouge_scores['rouge-l']['f']
    }
```

**Dependencies**: anthropic, nltk, rouge

### Week 12: Integration & Documentation

**Tasks**:
- [ ] Integrate all Phase 4 components: NLP → RAG → Knowledge Graph → Causal → Generation
- [ ] Create end-to-end demo: query accident patterns → retrieve reports → analyze causality → generate summary
- [ ] Build Phase 4 dashboard page: RAG chatbot, KG explorer, causal diagrams
- [ ] Document all APIs and workflows
- [ ] Performance optimization: caching, batch processing
- [ ] Security audit: sanitize inputs, rate limiting
- [ ] Prepare for Phase 5: containerization, deployment plan

**Deliverables**:
- Integrated AI system (end-to-end workflow)
- Phase 4 dashboard with 4 sub-pages
- Complete documentation (API reference, user guide)

**Success Metrics**:
- End-to-end query latency: <5 seconds
- Dashboard supports 10+ concurrent users
- Documentation completeness: 100% API coverage

**Sprint 4 Total Hours**: 80 hours

---

## Phase 4 Deliverables Summary

1. **NLP Pipeline**: spaCy preprocessing, SafeAeroBERT (87-91% accuracy), custom NER, BERTopic
2. **RAG System**: 10K+ vectorized reports, FAISS/Chroma, Claude/GPT-4 integration, precision@5 > 0.7
3. **Knowledge Graph**: Neo4j with 50K+ entities, 100K+ relationships, PageRank, community detection
4. **Causal Inference**: DoWhy models, 10+ validated causal effects
5. **Report Generation**: LLM-powered summaries, BLEU > 0.30, ROUGE-L > 0.40

## Testing Checklist

- [ ] NLP pipeline processes 10K narratives without errors
- [ ] SafeAeroBERT achieves 87%+ accuracy on test set
- [ ] RAG retrieval precision@5 > 0.70 (manual evaluation)
- [ ] Knowledge graph has 50K+ nodes, 100K+ edges
- [ ] Causal models pass refutation tests
- [ ] Generated reports evaluated by aviation expert (quality check)
- [ ] All APIs documented and tested

## Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| SafeAeroBERT accuracy | 87-91% | Test set evaluation |
| RAG precision@5 | >0.70 | Manual evaluation (100 queries) |
| RAG recall@10 | >0.80 | Manual evaluation |
| Knowledge graph size | 50K+ nodes | Neo4j query: MATCH (n) RETURN count(n) |
| Graph connectivity | 80%+ nodes connected | Graph analysis |
| Causal effects validated | 10+ significant | DoWhy + domain expert review |
| Report generation BLEU | >0.30 | Compare to NTSB reports |
| Report generation ROUGE-L | >0.40 | Compare to NTSB reports |

## Resource Requirements

**Infrastructure**:
- PostgreSQL (from Phase 1)
- Neo4j Community or AuraDB ($0-65/month)
- FAISS (local) or Chroma ($0-20/month)
- GPU optional for BERT training (can use CPU)

**External Services**:
- Anthropic Claude API: $100-500/month (depends on usage)
- OR OpenAI GPT-4: $100-500/month
- Alternative: Open-source LLMs (Llama 3, Mistral) - $0 but requires GPU

**Python Libraries**:
- **NLP**: spacy, transformers, datasets, bertopic
- **RAG**: sentence-transformers, faiss-cpu, anthropic
- **KG**: neo4j, networkx
- **Causal**: dowhy, scikit-learn
- **Evaluation**: nltk, rouge

**Estimated Budget**: $100-600/month (primarily LLM API costs)

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| BERT fine-tuning overfitting | Medium | Medium | Use dropout, early stopping, monitor val loss |
| LLM API costs exceed budget | Medium | High | Use caching, batch requests, consider open-source LLMs |
| Knowledge graph too sparse | Low | Medium | Focus on high-quality relationships, entity resolution |
| Causal inference assumptions violated | Medium | Medium | Use refutation tests, consult domain experts |
| Generated reports contain errors | High | High | Implement validation, human review, cite sources |

## Dependencies on Phase 1-3

- Clean narratives from PostgreSQL (Phase 1)
- Feature-engineered dataset (Phase 3)
- ML models for severity prediction (Phase 3)
- Airflow for scheduling (Phase 1)

## Next Phase

Upon completion, proceed to [PHASE_5_PRODUCTION.md](PHASE_5_PRODUCTION.md) for Kubernetes deployment, public API, and production launch.

---

**Last Updated**: November 2025
**Version**: 1.0
