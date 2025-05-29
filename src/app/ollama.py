import asyncio
import http.client
import json
import logging
import math
import os
import hashlib
from typing import List, Dict, Optional, Any
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import array

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# __Atom__ Base Class (polymorphing PyObject) and Data Classes
# ---------------------------------------------------------------------------

class __Atom__:
    __slots__ = ('_refcount',)
    
    def __init__(self) -> None:
        self._refcount: int = 1

    def incref(self) -> None:
        self._refcount += 1

    def decref(self) -> None:
        self._refcount -= 1
        if self._refcount <= 0:
            self.cleanup()

    def cleanup(self) -> None:
        # Override this method in subclasses to free resources
        pass

    @property
    def refcount(self) -> int:
        return self._refcount
@dataclass
class Document:
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    uuid: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest())

@dataclass
class EmbeddingConfig:
    dimensions: int = 768
    format_char: str = 'f'
    def get_format_char(self) -> str:
        return self.format_char

@dataclass
class MerkleNode:
    data: Dict[str, Any]
    children: List["MerkleNode"] = field(default_factory=list)
    hash: str = field(init=False)

    def __post_init__(self):
        self.hash = hashlib.sha256(json.dumps(self.data, sort_keys=True).encode()).hexdigest()

    def add_child(self, child: "MerkleNode") -> None:
        self.children.append(child)

@dataclass
class RuntimeState:
    merkle_root: Optional[MerkleNode] = None
    state_history: List[str] = field(default_factory=list)
class OllamaClient:
    def __init__(self, config: Optional['EmbeddingConfig'] = None):
        self.config = config or EmbeddingConfig()
        self.runtime_state = RuntimeState()
        self.ollama_client = OllamaClient()
        self.documents: List[Document] = []
        self.document_embeddings: Dict[str, array.array] = {}
        self.clusters: Dict[int, List[str]] = defaultdict(list)

    def __post__init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port

    async def generate_embedding(self, text: str, model: str = "nomic-embed-text") -> Optional[List[float]]:
        try:
            conn = http.client.HTTPConnection(self.host, self.port)
            request_data = {
                "model": model,
                "prompt": text
            }
            headers = {'Content-Type': 'application/json'}
            
            conn.request("POST", "/api/embeddings", json.dumps(request_data), headers)
            response = conn.getresponse()
            result = json.loads(response.read().decode())
            return result.get('embedding')
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None
        finally:
            conn.close()

    async def generate_response(self, prompt: str, model: str = "gemma:2b") -> str:
        try:
            conn = http.client.HTTPConnection(self.host, self.port)
            request_data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            headers = {'Content-Type': 'application/json'}
            
            conn.request("POST", "/api/generate", json.dumps(request_data), headers)
            response = conn.getresponse()
            result = json.loads(response.read().decode())
            return result.get('response', '')
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"Error generating response: {str(e)}"
        finally:
            conn.close()

    async def add_document(self, content: str, metadata: Optional[Dict] = None) -> Optional['Document']:
        try:
            embedding = await self.ollama_client.generate_embedding(content)
            if embedding:
                doc = Document(content=content, embedding=embedding, metadata=metadata)
                self.documents.append(doc)
                
                # Store embedding as array
                self.document_embeddings[doc.uuid] = array.array(
                    self.config.get_format_char(), 
                    embedding
                )
                
                # Assign to cluster
                cluster_id = self._assign_to_cluster(doc.uuid)
                self.clusters[cluster_id].append(doc.uuid)
                
                # Update Merkle tree
                await self._update_merkle_state()
                
                return doc
        except Exception as e:
            logger.error(f"Error adding document: {e}")
        return None

    def _assign_to_cluster(self, doc_uuid: str) -> int:
        if not self.clusters:
            return 0
        embedding = self.document_embeddings[doc_uuid]
        best_cluster = 0
        best_similarity = -1
        for cluster_id, doc_uuids in self.clusters.items():
            if doc_uuids:
                cluster_embedding = self._get_cluster_centroid(cluster_id)
                similarity = self._cosine_similarity(embedding, cluster_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id
        return best_cluster

    def _get_cluster_centroid(self, cluster_id: int) -> array.array:
        doc_uuids = self.clusters[cluster_id]
        if not doc_uuids:
            return array.array(self.config.get_format_char(), [0.0] * self.config.dimensions)
        embeddings = [self.document_embeddings[uuid] for uuid in doc_uuids]
        centroid = array.array(self.config.get_format_char(), [0.0] * self.config.dimensions)
        for emb in embeddings:
            for i in range(len(centroid)):
                centroid[i] += emb[i]
        for i in range(len(centroid)):
            centroid[i] /= len(embeddings)
        return centroid

    def _cosine_similarity(self, v1: array.array, v2: array.array) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(x * x for x in v1))
        norm2 = math.sqrt(sum(x * x for x in v2))
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    async def _update_merkle_state(self):
        system_state = {
            'timestamp': datetime.utcnow().isoformat(),
            'document_count': len(self.documents),
            'cluster_count': len(self.clusters),
            'config': self.config.__dict__
        }
        # Create new Merkle node for current state
        state_node = MerkleNode(system_state)
        for doc in self.documents:
            doc_node = MerkleNode({
                'uuid': doc.uuid,
                'content': doc.content,
                'metadata': doc.metadata
            })
            state_node.add_child(doc_node)
        
        self.runtime_state.merkle_root = state_node
        self.runtime_state.state_history.append(state_node.hash)
        await self._save_state()

    def _load_latest_previous_state(self):
        path = Path('states')
        if not path.exists():
            return None
        latest_state = None
        for state_file in path.glob('**/*.json'):
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                if not latest_state or state_data['timestamp'] > latest_state['timestamp']:
                    latest_state = state_data
        return latest_state

    def _generate_navigation_data(self, previous_state):
        return {
            'previous_state_hash': previous_state['root_hash'] if previous_state else None,
            'current_state_hash': self.runtime_state.merkle_root.hash
        }

    def _generate_index_data(self):
        return {
            'document_count': len(self.documents),
            'cluster_count': len(self.clusters)
        }

    def _calculate_state_deltas(self, previous_state):
        if not previous_state:
            return []
        current_state = self.runtime_state.merkle_root
        previous_state_node = MerkleNode(previous_state)

        def traverse(node1, node2, acc=None):
            if acc is None:
                acc = []
            if node1.hash != node2.hash:
                acc.append({
                    'type': 'update',
                    'path': node1.path, 
                })
            else:
                for child1, child2 in zip(node1.children, node2.children):
                    traverse(child1, child2, acc)
            return acc
        return traverse(current_state, previous_state_node)

    async def _save_state(self):
        previous_state = self._load_latest_previous_state()
        state_data = {
            'root_hash': self.runtime_state.merkle_root.hash,
            'parent_hash': previous_state['root_hash'] if previous_state else None,
            'version': '0.1.0',
            'timestamp': datetime.utcnow().isoformat(),
            'state_sequence': len(self.runtime_state.state_history),
            'merkle_metadata': self._generate_merkle_metadata(),
            'navigation': self._generate_navigation_data(previous_state),
            'index': self._generate_index_data(),
            'state_deltas': self._calculate_state_deltas(previous_state),
            # 'performance_metrics': self._collect_performance_metrics(),
            'documents': [...],
            'embeddings': {...},
            'clusters': {...},
            'state_history': self.runtime_state.state_history
        }
        # Save with nibble-wise organization
        path = Path('states') / self.runtime_state.merkle_root.hash[:2] / self.runtime_state.merkle_root.hash[2:4]
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"{self.runtime_state.merkle_root.hash}.json", 'w') as f:
            json.dump(state_data, f, indent=2)
    def _generate_merkle_metadata(self):
        def traverse_tree(node, level=0, acc=None):
            if acc is None:
                acc = defaultdict(list)
            acc[f"level_{level}"].append(node.hash)
            for child in node.children:
                traverse_tree(child, level + 1, acc)
            return acc
        node_references = traverse_tree(self.runtime_state.merkle_root)
        return {
            'tree_height': len(node_references),
            'total_nodes': sum(len(nodes) for nodes in node_references.values()),
            'node_references': dict(node_references)
        }

    async def query(self, query_text: str, top_k: int = 3) -> Dict:
        try:
            query_embedding = await self.ollama_client.generate_embedding(query_text)
            if not query_embedding:
                return {'error': 'Failed to generate query embedding'}
            query_array = array.array(self.config.get_format_char(), query_embedding)
            similarities = []
            for doc in self.documents:
                doc_embedding = self.document_embeddings[doc.uuid]
                similarity = self._cosine_similarity(query_array, doc_embedding)
                similarities.append((doc, similarity))
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_docs = similarities[:top_k]
            context = "\n".join([doc.content for doc, _ in top_docs])
            prompt = f"Context:\n{context}\n\nQuery: {query_text}\n\nResponse:"
            response = await self.ollama_client.generate_response(prompt)
            return {
                'query': query_text,
                'response': response,
                'similar_documents': [
                    {
                        'content': doc.content,
                        'similarity': score,
                        'metadata': doc.metadata
                    }
                    for doc, score in top_docs
                ]
            }
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {'error': str(e)}

# ---------------------------------------------------------------------------
# Helper Classes
# ---------------------------------------------------------------------------

async def main_feedback_loop(system: EnhancedRuntimeSystem) -> None:
    feedback_loop = FeedbackLoop(system)
    scores = await feedback_loop.evaluate_documents()
    await feedback_loop.apply_feedback(scores)

class Cache:
    def __init__(self, file_name: str = '.request_cache.json') -> None:
        self.file_name = file_name
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.file_name):
            with open(self.file_name, 'r') as f:
                return json.load(f)
        return {}

    def save_cache(self) -> None:
        with open(self.file_name, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = value
        self.save_cache()

class SyntaxKernel:
    def __init__(self, model_host: str, model_port: int) -> None:
        self.model_host = model_host
        self.model_port = model_port
        self.cache = Cache()

    async def fetch_from_api(self, path: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache_key = hashlib.sha256(json.dumps(data).encode()).hexdigest()
        cached_response = self.cache.get(cache_key)
        if cached_response:
            logger.info(f"Cache hit for {cache_key}")
            return cached_response
        logger.info(f"Querying API for {cache_key}")
        conn = http.client.HTTPConnection(self.model_host, self.model_port)
        try:
            headers = {'Content-Type': 'application/json'}
            conn.request("POST", path, json.dumps(data), headers)
            response = conn.getresponse()
            response_data = response.read().decode('utf-8')
            # Assume responses are newline-separated JSON objects
            json_objects = [json.loads(line) for line in response_data.strip().split('\n') if line.strip()]
            if json_objects:
                result = json_objects[-1]
                self.cache.set(cache_key, result)
                return result
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
        finally:
            conn.close()
        return None

    async def analyze_token(self, token: str) -> str:
        if len(token.split()) > 5:
            response = await self.fetch_from_api("/api/analyze", {"model": "gemma2", "query": token})
            return response.get('response', '') if response else "Analysis unavailable."
        return token