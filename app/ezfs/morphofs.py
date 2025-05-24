#!/usr/bin/env python3
"""
Knowledge Management System built on Morphological Source Code concepts.
This system leverages QuantumMemoryFS for storage and __Atom__ for knowledge representation.
It organizes unstructured pedagogical and architectural content into a queryable, evolvable system.
"""

import os
import re
import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, TypeVar, Generic
from dataclasses import dataclass, field

# Type variables for the FrameModel
T = TypeVar("T")
V = TypeVar("V") 
C = TypeVar("C")
class FrameModel(Generic[T, V, C]):
    """Represents a frame with delimited content."""
    
    def __init__(self, start_delimiter: str = "<<CONTENT>>", end_delimiter: str = "<<END_CONTENT>>"):
        self.start_delimiter = start_delimiter
        self.end_delimiter = end_delimiter
        self.content = ""
    
    def to_bytes(self) -> bytes:
        """Return the frame data as bytes."""
        return self.content.encode("utf-8")
    
    def parse_content(self, raw_content: str) -> str:
        """Parse the raw content between the defined delimiters."""
        start_index = raw_content.find(self.start_delimiter)
        end_index = raw_content.rfind(self.end_delimiter)
        if start_index == -1 or end_index == -1 or start_index >= end_index:
            raise ValueError("Invalid content format: delimiters not found or mismatched.")
        # Extract and return the content between delimiters.
        return raw_content[start_index + len(self.start_delimiter):end_index]
    
    def validate_content(self, content: str) -> bool:
        """Validate that content is wrapped with the proper delimiters."""
        return content.startswith(self.start_delimiter) and content.endswith(self.end_delimiter)


class __Atom__(Generic[T, V, C]):
    """Self-referential unit that encapsulates a FrameModel."""
    
    def __init__(self, frame: FrameModel[T, V, C]):
        self.frame = frame
        self.source = frame.content
        self.last_updated = datetime.now()
        self.id = uuid.uuid4().hex
        self.metadata = {}
        self.tags = set()
        self.references = set()
        self.embeddings = None
    
    def quine(self) -> str:
        """Return a self-referential representation of this __Atom__."""
        return f"{self.__class__.__name__} (ID: {self.id}, last updated: {self.last_updated.isoformat()})\n{self.source}"
    
    def update(self, new_content: str) -> None:
        """Update the __Atom__'s content if it passes validation."""
        if self.frame.validate_content(new_content):
            self.frame.content = new_content
            self.source = new_content
            self.last_updated = datetime.now()
        else:
            raise ValueError("New content does not pass delimiter validation.")


@dataclass
class KnowledgeEntry(__Atom__[str, str, Any]):
    """
    A knowledge entry representing a piece of content with metadata.
    This extends __Atom__ to include specific fields for knowledge management.
    """
    title: str = ""
    content_type: str = "text/markdown"  # Default content type
    related_entries: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def extract_frontmatter(self) -> Dict[str, Any]:
        """Extract frontmatter from markdown content if present."""
        content = self.source
        frontmatter = {}
        
        # Check for YAML frontmatter (between --- markers)
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter_text = frontmatter_match.group(1)
            try:
                # Simple parsing of key-value pairs
                for line in frontmatter_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        frontmatter[key.strip()] = value.strip().strip('"\'')
                
                # Update title from frontmatter if available
                if 'name' in frontmatter:
                    self.title = frontmatter['name']
                
                # Update links
                if 'link' in frontmatter:
                    self.related_entries.append(frontmatter['link'])
                
                # Update links from linklist
                if 'linklist' in frontmatter:
                    links = frontmatter['linklist']
                    if isinstance(links, str):
                        # Parse list-like string
                        link_matches = re.findall(r'\[\[(.*?)\]\]', links)
                        self.related_entries.extend(link_matches)
            except Exception as e:
                print(f"Error parsing frontmatter: {e}")
        
        return frontmatter
    
    def extract_links(self) -> List[str]:
        """Extract wiki-style links [[Link]] from content."""
        links = re.findall(r'\[\[(.*?)\]\]', self.source)
        self.related_entries = list(set(self.related_entries + links))
        return links
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.source,
            "content_type": self.content_type,
            "last_updated": self.last_updated.isoformat(),
            "related_entries": list(self.related_entries),
            "keywords": self.keywords,
            "metadata": self.metadata,
            "tags": list(self.tags)
        }


class KnowledgeBase:
    """
    A knowledge base system that organizes and manages KnowledgeEntry objects.
    This uses QuantumMemoryFS-inspired concepts for versioning and state tracking.
    """
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or os.path.join(os.getcwd(), 'knowledge_base'))
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.index: Dict[str, List[str]] = {
            "title": [],      # Index by title
            "keywords": {},   # Index by keywords
            "tags": {},       # Index by tags
            "links": {}       # Index by links
        }
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for a content entry based on its hash."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
    
    def add_entry(self, content: str, title: str = None, content_type: str = "text/markdown") -> str:
        """Add a new knowledge entry to the base."""
        # Create a delimited frame
        frame = FrameModel()
        frame.content = f"<<CONTENT>>{content}<<END_CONTENT>>"
        
        # Create knowledge entry
        entry = KnowledgeEntry(frame=frame, title=title or "Untitled", content_type=content_type)
        
        # Extract metadata
        entry.extract_frontmatter()
        entry.extract_links()
        
        # Generate ID if not already set 
        if not entry.id:
            entry.id = self._generate_id(content)
        
        # Store entry
        self.entries[entry.id] = entry
        
        # Update indices
        self._update_indices(entry)
        
        # Save to disk
        self._save_entry(entry)
        
        return entry.id
    
    def _update_indices(self, entry: KnowledgeEntry) -> None:
        """Update all indices with the entry."""
        # Title index
        self.index["title"].append(entry.id)
        
        # Keywords index
        for keyword in entry.keywords:
            if keyword not in self.index["keywords"]:
                self.index["keywords"][keyword] = []
            self.index["keywords"][keyword].append(entry.id)
        
        # Tags index
        for tag in entry.tags:
            if tag not in self.index["tags"]:
                self.index["tags"][tag] = []
            self.index["tags"][tag].append(entry.id)
        
        # Links index
        for link in entry.related_entries:
            if link not in self.index["links"]:
                self.index["links"][link] = []
            self.index["links"][link].append(entry.id)
    
    def _save_entry(self, entry: KnowledgeEntry) -> None:
        """Save an entry to disk with versioning."""
        # Create directory structure
        entry_dir = self.base_path / entry.id[:2] / entry.id[2:4] 
        entry_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current version
        entry_file = entry_dir / f"{entry.id}.json"
        with open(entry_file, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
        
        # Save version history (append timestamp)
        history_dir = entry_dir / "history"
        history_dir.mkdir(exist_ok=True)
        history_file = history_dir / f"{entry.id}_{entry.last_updated.isoformat().replace(':', '-')}.json"
        with open(history_file, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
    
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve an entry by ID."""
        if entry_id in self.entries:
            return self.entries[entry_id]
        
        # Try to load from disk if not in memory
        entry_dir = self.base_path / entry_id[:2] / entry_id[2:4]
        entry_file = entry_dir / f"{entry_id}.json"
        
        if entry_file.exists():
            try:
                with open(entry_file, 'r') as f:
                    data = json.load(f)
                
                frame = FrameModel()
                frame.content = f"<<CONTENT>>{data['content']}<<END_CONTENT>>"
                
                entry = KnowledgeEntry(
                    frame=frame,
                    title=data.get('title', 'Untitled'),
                    content_type=data.get('content_type', 'text/markdown')
                )
                entry.id = data['id']
                entry.last_updated = datetime.fromisoformat(data['last_updated'])
                entry.related_entries = data.get('related_entries', [])
                entry.keywords = data.get('keywords', [])
                entry.metadata = data.get('metadata', {})
                entry.tags = set(data.get('tags', []))
                
                self.entries[entry_id] = entry
                return entry
            except Exception as e:
                print(f"Error loading entry {entry_id}: {e}")
                return None
        
        return None
    
    def search(self, query: str, context: str = None) -> List[KnowledgeEntry]:
        """
        Search for entries matching a query.
        
        Args:
            query: The search query
            context: Optional context to narrow search (e.g., 'tags', 'keywords')
            
        Returns:
            A list of matching KnowledgeEntry objects
        """
        results = []
        
        # Simple text search implementation
        for entry_id, entry in self.entries.items():
            if query.lower() in entry.title.lower() or query.lower() in entry.source.lower():
                results.append(entry)
                continue
            
            # Check keywords and tags
            for keyword in entry.keywords:
                if query.lower() in keyword.lower():
                    results.append(entry)
                    break
            
            for tag in entry.tags:
                if query.lower() in tag.lower():
                    results.append(entry)
                    break
        
        return results
    
    def get_related(self, entry_id: str) -> List[KnowledgeEntry]:
        """Get entries related to a given entry."""
        entry = self.get_entry(entry_id)
        if not entry:
            return []
        
        related_ids = []
        for link_text in entry.related_entries:
            if link_text in self.index["links"]:
                related_ids.extend(self.index["links"][link_text])
        
        # Remove duplicates and self-reference
        related_ids = list(set(related_ids))
        if entry_id in related_ids:
            related_ids.remove(entry_id)
        
        # Load entries
        return [self.get_entry(rid) for rid in related_ids if self.get_entry(rid)]
    
    def build_graph(self) -> Dict[str, Any]:
        """
        Build a knowledge graph representation of the entire knowledge base.
        
        Returns:
            A dictionary with nodes and edges representing the knowledge graph
        """
        nodes = []
        edges = []
        
        # Create nodes for each entry
        for entry_id, entry in self.entries.items():
            nodes.append({
                "id": entry_id,
                "label": entry.title,
                "type": "entry"
            })
            
            # Create edges for related entries
            for related_text in entry.related_entries:
                # Find entries with this link text
                related_entries = []
                if related_text in self.index["links"]:
                    related_entries = self.index["links"][related_text]
                
                for related_id in related_entries:
                    if related_id != entry_id:  # Avoid self-loops
                        edges.append({
                            "source": entry_id,
                            "target": related_id,
                            "label": "related"
                        })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def import_markdown_file(self, file_path: str) -> str:
        """Import a markdown file into the knowledge base."""
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"File {file_path} does not exist")
        
        content = path.read_text(encoding="utf-8")
        title = path.stem  # Use filename as default title
        
        return self.add_entry(content, title=title, content_type="text/markdown")
    
    def bulk_import(self, directory: str, pattern: str = "*.md") -> List[str]:
        """Import all files matching pattern from a directory."""
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory {directory} does not exist")
        
        imported_ids = []
        for file_path in dir_path.glob(pattern):
            try:
                entry_id = self.import_markdown_file(str(file_path))
                imported_ids.append(entry_id)
                print(f"Imported {file_path} as {entry_id}")
            except Exception as e:
                print(f"Error importing {file_path}: {e}")
        
        return imported_ids


def main():
    """Demo of the knowledge base system."""
    kb = KnowledgeBase()
    
    # Add some sample entries
    quantum_id = kb.add_entry("""---
name: "Quantum Computing Basics"
tags: ["quantum", "computing", "theory"]
---
# Quantum Computing Basics

Quantum computing uses quantum-mechanical phenomena to perform computation.
Unlike classical computers, quantum computers use qubits which can represent
a superposition of states.

See also: [[Hilbert Space]], [[Quantum Gates]]
""")
    
    hilbert_id = kb.add_entry("""---
name: "Hilbert Space"
tags: ["mathematics", "quantum"]
---
# Hilbert Space

A Hilbert space is a vector space with an inner product that allows length
and angle to be measured. It generalizes Euclidean space and is used extensively
in quantum mechanics.

See also: [[Quantum Computing Basics]], [[Vector Spaces]]
""")
    
    # Retrieve and display entries
    quantum_entry = kb.get_entry(quantum_id)
    print(f"Entry: {quantum_entry.title}")
    print(f"Tags: {quantum_entry.tags}")
    print(f"Links: {quantum_entry.related_entries}")
    
    # Find related entries
    related = kb.get_related(quantum_id)
    print(f"\nEntries related to {quantum_entry.title}:")
    for entry in related:
        print(f"- {entry.title}")
    
    # Generate knowledge graph
    graph = kb.build_graph()
    print(f"\nKnowledge Graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")


if __name__ == "__main__":
    main()