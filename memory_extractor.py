"""
Memory Extractor - Async Insight Extraction Agent

Extracts structured memories from conversation exchanges using a cheap/fast LLM.
Runs asynchronously to avoid blocking the main conversation flow.

This is the "memory agent" that intelligently decides what to remember.
"""

import json
import os
import sys
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from token_tracker import tracker

# Will be set when Gemini client is available
_gemini_available = False
_genai_client = None
try:
    from google import genai
    _gemini_available = True
except ImportError:
    pass


@dataclass
class ExtractedMemory:
    """A memory extracted from conversation"""
    content: str
    importance: float
    memory_type: str  # fact, preference, decision, insight
    tags: List[str]
    confidence: float
    source_context: str


EXTRACTION_PROMPT = """
You are a memory extraction agent. You act as the Mirokai Robot's subconscious.
Analyze this conversation exchange and extract NEW factual information worth remembering.

Extract memories as JSON. Preserve SPECIFIC details - these are critical for accurate recall.

MUST preserve:
- DATES and TIMES - CONVERT relative dates to absolute when possible! All memories should be
  - If message says "[May 8, 2023] yesterday" → store as "May 7, 2023"
  - If message says "[May 25, 2023] last year" → store as "2022"
  - If message says "[June 9, 2023] next month" → store as "July 2023"
- NAMES of people, places, organizations  
- SPECIFIC facts (identities, relationships, events, activities)
- Numbers, quantities, durations

DO extract:
- Who someone is (identity, profession, relationships)
- What someone did or experienced (events, activities)
- When things happened (dates, timeframes)
- Preferences, beliefs, values
- Decisions and plans
- Biographical details

DON'T extract:
- Generic greetings or small talk
- Vague acknowledgments without substance
- Information already present in the CONTEXT section above.

CONTEXT (Injected Memories):
{injected_memories}
(DO NOT create new memories from the context above - this is what the robot already knows.)

CONVERSATION HISTORY:
{conversation_history}
(This is the past conversation leading up to the current exchange. This is what the robot already knows.)

EXCHANGE:
User message : {user_message}
(New information here should be added to memory)

Robot response : {assistant_response}
(New information here should be added to memory)

TO UPDATE AN EXISTING MEMORY:
If the user or robot explicitly updates, corrects, or expands on a memory shown in CONTEXT, use "modifies": ID.
Example: If [ID: 3] says "User likes apples" and user says "I actually hate apples", return:
{{
  "content": "User hates apples",
  "modifies": 3,
  "type": "preference"
}}



Return JSON only, no markdown:
{{
  "should_remember": true/false,
  "memories": [
    {{
      "content": "specific factual statement preserving names, dates, and details",
      "type": "fact|event|preference|identity|relationship",
      "importance": 0.0-1.0,
      "tags": ["relevant", "tags"],
      "confidence": 0.0-1.0,
      "modifies": optional_integer_id
    }}
  ],
  "reasoning": "brief explanation"
}}

IMPORTANT: Be specific! "Alex is a software developer" not "Alex works in tech".
Include dates: "Sarah joined the hiking club on March 8, 2025" not "Sarah joined a club".

If nothing substantive, return should_remember: false with empty memories array."""


class MemoryExtractor:
    """
    Extracts memories from conversation exchanges asynchronously.
    
    Uses a cheap LLM (Gemini Flash) to analyze conversations
    and extract structured information worth remembering.
    """
    
    def __init__(self, memory_system=None, model: str = "gemini-2.0-flash-lite"):
        """
        Initialize the memory extractor.
        
        Args:
            memory_system: VectorMemory instance to store extracted memories
            model: Model to use for extraction (default: Gemini Flash for cost efficiency)
        """
        self.memory_system = memory_system
        self.model_name = model
        self.client = None
        
        # Extraction queue for async processing
        self.extraction_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        
        # Message queue for UI output (avoids printing after prompt)
        self.message_queue = queue.Queue()
        
        # Statistics
        self.stats = {
            "exchanges_processed": 0,
            "memories_extracted": 0,
            "extraction_errors": 0,
            "last_extraction": None
        }
        
        # Activity log for visualizer
        self.activity_log_path = Path("vector_memory") / "activity.jsonl"
        
        # Initialize client if available
        if _gemini_available:
            try:
                # Try to get API key from environment
                api_key = os.environ.get("GEMINI_API_KEY")
                if api_key:
                    self.client = genai.Client(api_key=api_key)
                    print(f"🧠 Memory Extractor initialized with Gemini ({model})")
                else:
                    print("⚠️  GEMINI_API_KEY not found in environment")
                    print("   Memory extraction will use fallback heuristics")
            except Exception as e:
                print(f"⚠️  Could not initialize Gemini client: {e}")
                print("   Memory extraction will use fallback heuristics")
        else:
            print("⚠️  google-genai SDK not installed. Using fallback extraction.")
            print("   Install with: pip install google-genai")
    
    def start_worker(self):
        """Start the background extraction worker"""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._extraction_worker, daemon=True)
        self.worker_thread.start()
        print("🔄 Memory extraction worker started")
    
    def stop_worker(self):
        """Stop the background extraction worker"""
        self.running = False
        if self.worker_thread:
            # Put a sentinel to wake up the worker
            self.extraction_queue.put(None)
            self.worker_thread.join(timeout=2.0)
    
    def _extraction_worker(self):
        """Background worker that processes extraction queue"""
        while self.running:
            try:
                item = self.extraction_queue.get(timeout=1.0)
                
                if item is None:  # Sentinel to stop
                    break
                
                user_msg, assistant_msg, injected_mems, history, i_map, callback = item
                
                try:
                    memories = self._extract_memories_sync(user_msg, assistant_msg, injected_mems, history, i_map)
                    
                    # Store extracted memories
                    for memory in memories:
                        self._store_memory(memory)
                    
                    if callback:
                        callback(memories)
                        
                except Exception as e:
                    self.stats["extraction_errors"] += 1
                    self._queue_message(f"MemMan: ⚠️  Extraction error: {e}")
                
                self.extraction_queue.task_done()
                
            except queue.Empty:
                continue
    
    def extract_async(self, user_message: str, assistant_response: str, 
                     injected_memories: str = None, conversation_history: List[Dict[str, str]] = None, 
                     injection_map: Dict[int, str] = None, callback: callable = None):
        """
        Queue an exchange for async memory extraction.
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            callback: Optional callback with extracted memories
        """
        # Start worker if not running
        if not self.running:
            self.start_worker()
        
        # Queue for extraction
        self.extraction_queue.put((user_message, assistant_response, injected_memories, conversation_history, injection_map, callback))
    
    def _extract_memories_sync(self, user_message: str,
                               assistant_response: str,
                               injected_memories: str = None,
                               conversation_history: List[Dict[str, str]] = None,
                               injection_map: Dict[int, str] = None) -> List[ExtractedMemory]:
        """
        Synchronously extract memories from an exchange.
        
        Returns:
            List of ExtractedMemory objects
        """
        self.stats["exchanges_processed"] += 1
        self.stats["last_extraction"] = datetime.now()
        
        # Use LLM extraction if available
        if self.client:
            return self._extract_with_llm(user_message, assistant_response, injected_memories, conversation_history, injection_map)
        else:
            return self._extract_with_heuristics(user_message, assistant_response)
    
    def _extract_with_llm(self, user_message: str, 
                          assistant_response: str,
                          injected_memories: str = None,
                          conversation_history: List[Dict[str, str]] = None,
                          injection_map: Dict[int, str] = None) -> List[ExtractedMemory]:
        """Extract memories using Gemini LLM"""
        try:
            injected_context = injected_memories if injected_memories else "No previous context injected."
            
            # Format history
            history_str = "No previous history."
            if conversation_history:
                history_lines = []
                for msg in conversation_history:
                    role_name = "Robot" if msg.get("role") in ["model", "assistant"] else "User"
                    history_lines.append(f"{role_name}: {msg.get('content', '')}")
                history_str = "\n".join(history_lines)

            prompt = EXTRACTION_PROMPT.format(
                user_message=user_message[:2000],  # Limit length
                assistant_response=assistant_response[:2000],
                injected_memories=injected_context[:3000], # Limit context length
                conversation_history=history_str[:5000] # Limit history length
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            response_text = response.text
            
            # Track token usage for MemMan
            tracker.record_memman_usage(response.usage_metadata, self.model_name)
            
            # Parse response
            try:
                # Try to find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    data = json.loads(json_str)
                else:
                    return []
            except json.JSONDecodeError:
                return []
            
            if not data.get("should_remember", False):
                return []
            
            memories = []
            raw_memories = data.get("memories", [])
            
            # # SAFEGUARD: Volume Warning
            # if len(raw_memories) > 10:
            #     self._queue_message(f"MemMan: ⚠️  Warning: High volume extraction ({len(raw_memories)} items). Checking context...")
                
            # # SAFEGUARD: Hard Limit
            # if len(raw_memories) > 15:
            #     self._queue_message(f"MemMan: 🛑  Safety Limit: Truncating {len(raw_memories)} -> 15 memories")
            #     raw_memories = raw_memories[:15]
            
            for mem_data in raw_memories:
                memory = ExtractedMemory(
                    content=mem_data.get("content", ""),
                    importance=float(mem_data.get("importance", 0.5)),
                    memory_type=mem_data.get("type", "insight"),
                    tags=mem_data.get("tags", []),
                    confidence=float(mem_data.get("confidence", 0.7)),
                    source_context="conversation"
                )
                
                # Handle modification request
                mod_id_raw = mem_data.get("modifies")
                mod_id = None
                
                if mod_id_raw is not None:
                    # STRICT PARSING Logic
                    try:
                        # Case 1: Clean integer
                        mod_id = int(str(mod_id_raw))
                    except ValueError:
                        # Case 2: Dirty string (e.g., "ID: 3", "#3")
                        import re
                        digits = re.findall(r'\d+', str(mod_id_raw))
                        if digits:
                            if len(digits) > 1 or len(re.sub(r'\d', '', str(mod_id_raw)).strip()) > 0:
                                self._queue_message(f"MemMan: ⚠️  Warning: ambiguous ID format '{mod_id_raw}'. Using extracted number '{digits[0]}'.")
                            mod_id = int(digits[0])
                        else:
                             self._queue_message(f"MemMan: 🛑 Error: Invalid modification ID '{mod_id_raw}'. Ignoring modification request.")
                             mod_id = None

                if mod_id and injection_map:
                    real_id = injection_map.get(mod_id)
                    if real_id:
                        setattr(memory, 'modifies_memory_id', real_id)



                
                if memory.content:
                    memories.append(memory)
            
            self.stats["memories_extracted"] += len(memories)
            return memories
            
        except Exception as e:
            self._queue_message(f"MemMan: ⚠️  LLM extraction failed: {e}")
            return self._extract_with_heuristics(user_message, assistant_response)
    
    def _extract_with_heuristics(self, user_message: str, 
                                  assistant_response: str) -> List[ExtractedMemory]:
        """Fallback heuristic-based extraction when LLM unavailable"""
        memories = []
        
        combined_text = f"{user_message} {assistant_response}".lower()
        
        # Preference detection
        preference_indicators = [
            "i prefer", "i like", "i always", "i usually", "i want",
            "please always", "please don't", "i don't like"
        ]
        
        for indicator in preference_indicators:
            if indicator in user_message.lower():
                # Extract the sentence containing the preference
                sentences = user_message.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        memory = ExtractedMemory(
                            content=sentence.strip(),
                            importance=0.7,
                            memory_type="preference",
                            tags=["preference", "user"],
                            confidence=0.6,
                            source_context="heuristic_extraction"
                        )
                        memories.append(memory)
                        break
        
        # Decision detection
        decision_indicators = [
            "decided to", "going to", "will use", "chose to",
            "let's go with", "we'll use"
        ]
        
        for indicator in decision_indicators:
            if indicator in combined_text:
                for sentence in (user_message + " " + assistant_response).split('.'):
                    if indicator in sentence.lower():
                        memory = ExtractedMemory(
                            content=f"Decision: {sentence.strip()}",
                            importance=0.6,
                            memory_type="decision",
                            tags=["decision"],
                            confidence=0.5,
                            source_context="heuristic_extraction"
                        )
                        memories.append(memory)
                        break
        
        self.stats["memories_extracted"] += len(memories)
        return memories
    
    def _log_activity(self, action: str, details: Dict[str, Any]):
        """Log activity for visualizer to display"""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                **details
            }
            with open(self.activity_log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            
            # Keep log file reasonable size (last 100 entries)
            try:
                with open(self.activity_log_path, 'r') as f:
                    lines = f.readlines()
                if len(lines) > 100:
                    with open(self.activity_log_path, 'w') as f:
                        f.writelines(lines[-100:])
            except Exception:
                pass
        except Exception:
            pass  # Don't fail on logging errors

    def _store_memory(self, memory: ExtractedMemory):
        """Store an extracted memory in the memory system"""
        if not self.memory_system:
            return
        
        # Adjust importance by confidence
        adjusted_importance = memory.importance * memory.confidence
        
        # Add memory type to tags
        tags = memory.tags.copy()
        if memory.memory_type not in tags:
            tags.append(memory.memory_type)
        
        if memory.memory_type not in tags:
            tags.append(memory.memory_type)
        


        try:
            # Check if this is a modification of an existing memory
            if hasattr(memory, 'modifies_memory_id') and memory.modifies_memory_id:
                real_id = memory.modifies_memory_id
                
                # Retrieve old content for logging purpose
                old_content = "Unknown"
                if real_id in self.memory_system.memories:
                    old_content = self.memory_system.memories[real_id].content

                success = self.memory_system.update_memory(
                    memory_id=real_id,
                    content=memory.content,
                    tags=tags, # Use the potentially updated tags
                    importance=adjusted_importance # Use the adjusted importance
                )
                if success:
                     self._queue_message(f"MemMan: ✏️  Modified : '{old_content}' -> '{memory.content}'")
                     
                     # Log activity for visualizer
                     stored_memory = self.memory_system.memories.get(real_id)
                     self._log_activity("memory_updated", {
                         "memory_id": real_id,
                         "content": memory.content[:80],
                         "old_content": old_content[:80],
                         "importance": adjusted_importance,
                         "type": memory.memory_type,
                         "tags": tags,
                         "access_count": stored_memory.access_count if stored_memory else 1
                     })
                     return
                else:
                     self._queue_message(f"MemMan: ⚠️ Could not update memory [{real_id[:8]}], falling back to create.")

            memory_id = self.memory_system.store_memory(
                content=memory.content,
                importance=adjusted_importance,
                tags=tags,
                context={
                    "source": memory.source_context,
                    "type": memory.memory_type,
                    "confidence": memory.confidence,
                    "extracted_at": datetime.now().isoformat()
                }
            )
            # Force save so visualizer can see new memories immediately
            self.memory_system.force_save()
            
            # Check if this was an update (access_count > 1) or new creation
            stored_memory = self.memory_system.memories.get(memory_id)
            was_update = stored_memory and stored_memory.access_count > 1
            
            # Log activity for visualizer
            self._log_activity("memory_updated" if was_update else "memory_stored", {
                "memory_id": memory_id,
                "content": memory.content[:80],
                "importance": adjusted_importance,
                "type": memory.memory_type,
                "tags": tags,
                "access_count": stored_memory.access_count if stored_memory else 1
            })
            
            if was_update:
                # Get update stats from memory system
                stats = getattr(self.memory_system, '_last_update_stats', {})
                acc = stats.get('access_count', stored_memory.access_count if stored_memory else 1)
                old_imp = stats.get('old_importance', adjusted_importance)
                new_imp = stats.get('new_importance', adjusted_importance)
                self._queue_message(f"MemMan: 🔄 Reinforced (acc:{acc}, imp:{old_imp:.2f}→{new_imp:.2f}): {memory.content}")
            else:
                self._queue_message(f"MemMan: 💾 New (imp:{adjusted_importance:.2f}): {memory.content}")
        except Exception as e:
            self._log_activity("store_error", {"error": str(e), "content": memory.content[:50]})
            self._queue_message(f"MemMan: ⚠️  Failed to store memory: {e}")
    
    def _queue_message(self, message: str):
        """Print message above current line using terminal escape codes"""
        # \r = move to start of line, \033[K = clear line
        # Print message, newline, then reprint the prompt
        sys.stdout.write(f"\r\033[K{message}\n\033[1;36mYou:\033[0m ")
        sys.stdout.flush()
    
    def get_pending_messages(self) -> List[str]:
        """Get and clear all pending messages (legacy - messages now print immediately)"""
        messages = []
        while not self.message_queue.empty():
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages
    
    def print_pending_messages(self):
        """Print all pending messages (legacy - messages now print immediately)"""
        for msg in self.get_pending_messages():
            print(msg)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            **self.stats,
            "queue_size": self.extraction_queue.qsize(),
            "worker_running": self.running
        }


def create_memory_extractor(memory_system=None) -> MemoryExtractor:
    """Factory function to create a memory extractor"""
    return MemoryExtractor(memory_system=memory_system)


if __name__ == "__main__":
    # Test the extractor
    print("Testing Memory Extractor")
    print("=" * 40)
    
    extractor = MemoryExtractor()
    
    # Test extraction
    test_user = "I prefer dark mode in all my applications and I'm working on a project called ProjectX"
    test_assistant = "I'll keep that preference in mind. For ProjectX, let me know what you need help with."
    
    memories = extractor._extract_memories_sync(test_user, test_assistant)
    
    print(f"\nExtracted {len(memories)} memories:")
    for mem in memories:
        print(f"  - [{mem.memory_type}] {mem.content}")
        print(f"    Importance: {mem.importance}, Confidence: {mem.confidence}")
    
    print(f"\nStats: {extractor.get_stats()}")
