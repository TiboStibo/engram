"""
Context Window Manager

Efficiently manages token allocation and memory integration within context window limits.
Optimizes information density while respecting LLM context constraints.

Key Features:
- Dynamic token allocation based on context importance
- Memory compression and summarization for space efficiency
- Priority-based memory selection and ordering
- Real-time context window monitoring
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class TokenBudget:
    """Manages token allocation within context window"""
    total_tokens: int
    used_tokens: int = 0
    reserved_tokens: int = 0  # For system prompts, etc.

    @property
    def available_tokens(self) -> int:
        """Get available tokens for memory integration"""
        return max(0, self.total_tokens - self.used_tokens - self.reserved_tokens)

    def allocate_tokens(self, requested_tokens: int) -> int:
        """Allocate tokens, returns actual allocated amount"""
        available = self.available_tokens
        allocated = min(requested_tokens, available)
        self.used_tokens += allocated
        return allocated

    def reserve_tokens(self, tokens: int):
        """Reserve tokens for fixed content"""
        self.reserved_tokens += tokens


@dataclass
class CompressedMemory:
    """Memory optimized for context window inclusion"""
    original_id: str
    content: str
    importance: float
    priority: float  # Calculated priority score
    token_count: int
    compression_ratio: float  # How much it was compressed
    summary: Optional[str] = None  # Compressed version if applicable

    def get_optimal_content(self) -> str:
        """Get the best content version for available space"""
        return self.summary if self.summary and len(self.summary) < len(self.content) else self.content


class ContextWindowManager:
    """
    Intelligent context window management for memory integration.

    Optimizes information density by:
    - Compressing memories when necessary
    - Prioritizing high-value content
    - Dynamic allocation based on context needs
    """

    def __init__(self, max_tokens: int = 8000, compression_enabled: bool = True):
        self.token_budget = TokenBudget(total_tokens=max_tokens)
        self.compression_enabled = compression_enabled

        # Token estimation (conservative)
        self.chars_per_token = 4.0
        self.compression_ratios = {
            'high': 0.3,    # Aggressive compression for critical space
            'medium': 0.5,  # Moderate compression
            'low': 0.7     # Light compression
        }

        # Priority weights for different memory attributes
        self.priority_weights = {
            'importance': 0.4,
            'relevance': 0.3,
            'recency': 0.15,
            'usage': 0.15
        }

    def set_context_usage(self, used_tokens: int, reserved_tokens: int = 1000):
        """Set current context usage (conversation + system prompts)"""
        self.token_budget.used_tokens = used_tokens
        self.token_budget.reserved_tokens = reserved_tokens

    def optimize_memories_for_context(self, memories: List[Any],
                                    available_tokens: Optional[int] = None) -> List[CompressedMemory]:
        """
        Optimize a list of memories for context inclusion

        Args:
            memories: List of ContextMemory or similar objects
            available_tokens: Override automatic token calculation

        Returns:
            List of CompressedMemory objects optimized for space
        """
        if not memories:
            return []

        # Calculate available tokens
        if available_tokens is None:
            available_tokens = self.token_budget.available_tokens

        # Convert to CompressedMemory format and calculate priorities
        compressed_memories = []
        for memory in memories:
            compressed = self._compress_memory(memory)
            compressed.priority = self._calculate_priority(memory)
            compressed_memories.append(compressed)

        # Sort by priority (highest first)
        compressed_memories.sort(key=lambda x: x.priority, reverse=True)

        # Select optimal subset within token budget
        selected_memories = []
        used_tokens = 0

        for memory in compressed_memories:
            optimal_content = memory.get_optimal_content()
            content_tokens = len(optimal_content) // self.chars_per_token

            if used_tokens + content_tokens <= available_tokens:
                selected_memories.append(memory)
                used_tokens += content_tokens
            else:
                # Try compressed version if original doesn't fit
                if memory.summary and memory.summary != optimal_content:
                    summary_tokens = len(memory.summary) // self.chars_per_token
                    if used_tokens + summary_tokens <= available_tokens:
                        # Update to use summary
                        memory.content = memory.summary
                        memory.token_count = summary_tokens
                        selected_memories.append(memory)
                        used_tokens += summary_tokens

        return selected_memories

    def _compress_memory(self, memory) -> CompressedMemory:
        """Compress memory content for efficient context usage"""
        original_content = memory.content
        original_tokens = len(original_content) // self.chars_per_token

        compressed = CompressedMemory(
            original_id=memory.memory_id,
            content=original_content,
            importance=getattr(memory, 'importance', 0.5),
            priority=0.0,  # Will be calculated later
            token_count=original_tokens,
            compression_ratio=1.0
        )

        if self.compression_enabled and original_tokens > 50:  # Only compress longer memories
            compressed.summary = self._generate_summary(original_content)
            if compressed.summary:
                summary_tokens = len(compressed.summary) // self.chars_per_token
                compressed.compression_ratio = summary_tokens / original_tokens

        return compressed

    def _generate_summary(self, content: str) -> Optional[str]:
        """Generate a compressed summary of memory content"""
        # Simple extractive summarization - keep most important sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 2:
            return None  # Too short to summarize effectively

        # Score sentences by length and position (simple heuristic)
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Prefer longer sentences and those in the middle (often contain key info)
            position_bonus = 1.0 if 0 < i < len(sentences) - 1 else 0.7
            length_score = min(1.0, len(sentence) / 100)  # Cap at 100 chars
            score = length_score * position_bonus
            scored_sentences.append((score, sentence))

        # Select top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = scored_sentences[:2]  # Keep 2 most important

        # Reconstruct summary
        summary = '. '.join([sentence for _, sentence in top_sentences])
        if not summary.endswith('.'):
            summary += '.'

        return summary if len(summary) < len(content) * 0.7 else None

    def _calculate_priority(self, memory) -> float:
        """Calculate priority score for memory inclusion"""
        importance = getattr(memory, 'importance', 0.5)
        relevance = getattr(memory, 'relevance_score', 0.5)
        usage_count = getattr(memory, 'usage_count', 0)

        # Recency score (newer memories get slight preference)
        recency_score = 0.5  # Default for memories without timestamp
        if hasattr(memory, 'last_used') and memory.last_used:
            days_old = (datetime.now() - memory.last_used).days
            recency_score = max(0.1, 1 - days_old / 30)  # 30-day decay

        # Usage score (more used = higher priority)
        usage_score = min(1.0, usage_count / 10)  # Cap at 10 uses

        # Weighted combination
        priority = (
            self.priority_weights['importance'] * importance +
            self.priority_weights['relevance'] * relevance +
            self.priority_weights['recency'] * recency_score +
            self.priority_weights['usage'] * usage_score
        )

        return priority

    def format_optimized_context(self, memories: List[CompressedMemory]) -> str:
        """Format optimized memories for context integration"""
        if not memories:
            return ""

        # Group by priority levels
        high_priority = [m for m in memories if m.priority >= 0.7]
        medium_priority = [m for m in memories if 0.4 <= m.priority < 0.7]
        low_priority = [m for m in memories if m.priority < 0.4]

        sections = []

        if high_priority:
            sections.append("🔥 CRITICAL CONTEXT:")
            sections.extend([f"• {memory.get_optimal_content()}" for memory in high_priority])

        if medium_priority:
            if sections:
                sections.append("")
            sections.append("📝 RELEVANT INFORMATION:")
            sections.extend([f"• {memory.get_optimal_content()}" for memory in medium_priority])

        if low_priority:
            if sections:
                sections.append("")
            sections.append("💭 ADDITIONAL NOTES:")
            sections.extend([f"• {memory.get_optimal_content()}" for memory in low_priority])

        return "\n".join(sections)

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about context window usage"""
        return {
            "total_tokens": self.token_budget.total_tokens,
            "used_tokens": self.token_budget.used_tokens,
            "reserved_tokens": self.token_budget.reserved_tokens,
            "available_tokens": self.token_budget.available_tokens,
            "usage_percentage": (self.token_budget.used_tokens / self.token_budget.total_tokens) * 100,
            "compression_enabled": self.compression_enabled
        }

    def estimate_token_usage(self, text: str) -> int:
        """Estimate token count for a text string"""
        return int(len(text) / self.chars_per_token)


# Integration helper functions
def integrate_memories_into_context(
    memories: List[Any],
    context_window_manager: ContextWindowManager,
    current_context_tokens: int = 0
) -> str:
    """
    High-level function to integrate memories into context efficiently

    Args:
        memories: List of memory objects to integrate
        context_window_manager: Configured context manager
        current_context_tokens: Tokens already used by conversation

    Returns:
        Formatted context string with integrated memories
    """
    # Update context usage
    context_window_manager.set_context_usage(current_context_tokens)

    # Optimize memories for available space
    optimized_memories = context_window_manager.optimize_memories_for_context(memories)

    # Format for context integration
    integrated_context = context_window_manager.format_optimized_context(optimized_memories)

    return integrated_context


def create_default_context_manager(max_tokens: int = 8000) -> ContextWindowManager:
    """Create a context window manager with sensible defaults"""
    return ContextWindowManager(max_tokens=max_tokens, compression_enabled=True)


if __name__ == "__main__":
    # Test the context window manager
    manager = create_default_context_manager(max_tokens=4000)
    manager.set_context_usage(2000, 500)  # 2000 used, 500 reserved

    print("Context Window Stats:")
    print(json.dumps(manager.get_context_stats(), indent=2))

    # Test token estimation
    test_text = "This is a test memory about vector databases and their performance characteristics."
    tokens = manager.estimate_token_usage(test_text)
    print(f"\nEstimated tokens for test text: {tokens}")
    print(f"Available tokens: {manager.token_budget.available_tokens}")
