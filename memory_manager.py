#!/usr/bin/env python3
"""
Memory Manager - Interactive TUI for managing Engram memories.

A full-featured terminal UI for browsing, inspecting, editing, and archiving memories.
Built with Textual for rich interactivity.

Usage:
    python memory_manager.py                    # Default, uses ./vector_memory
    python memory_manager.py --memory-path /path/to/memory
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Static, DataTable, Input, Button, Label, TextArea
)
from textual.screen import ModalScreen
from textual.message import Message
from rich.text import Text


# Stub for loading memories without full engram import
class MemoryEntryStub:
    """Stub class for unpickling MemoryEntry objects"""
    def __init__(self):
        self.id = ""
        self.content = ""
        self.timestamp = datetime.now()
        self.importance = 0.5
        self.tags = []
        self.context = {}
        self.access_count = 0
        self.last_accessed = None
        self.related_memories = []
        self.embedding = []
        self.faiss_index = -1
        self.archived = False

    def __setstate__(self, state):
        self.__dict__.update(state)


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that can load MemoryEntry without full engram_pkg"""
    def find_class(self, module, name):
        if name == 'MemoryEntry':
            return MemoryEntryStub
        return super().find_class(module, name)


class ConfirmDialog(ModalScreen[bool]):
    """A confirmation dialog modal."""
    
    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, message: str, action: str = "confirm"):
        super().__init__()
        self.message = message
        self.action_name = action
    
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(self.message, id="dialog-message")
            with Horizontal(id="dialog-buttons"):
                yield Button("Yes (Y)", variant="error", id="yes-btn")
                yield Button("No (N)", variant="primary", id="no-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes-btn")
    
    def action_confirm(self) -> None:
        self.dismiss(True)
    
    def action_cancel(self) -> None:
        self.dismiss(False)


class EditMemoryScreen(ModalScreen[dict]):
    """Modal screen for editing a memory."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]
    
    def __init__(self, memory):
        super().__init__()
        self.memory = memory
    
    def compose(self) -> ComposeResult:
        with Container(id="edit-dialog"):
            yield Label(f"Editing Memory: {self.memory.id[:8]}...", id="edit-title")
            
            yield Label("Content:")
            yield TextArea(self.memory.content, id="edit-content")
            
            yield Label("Importance (0.0 - 1.0):")
            yield Input(str(self.memory.importance), id="edit-importance")
            
            yield Label("Tags (comma-separated):")
            yield Input(", ".join(self.memory.tags), id="edit-tags")
            
            with Horizontal(id="edit-buttons"):
                yield Button("Save (Ctrl+S)", variant="success", id="save-btn")
                yield Button("Cancel (Esc)", variant="error", id="cancel-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self.action_save()
        else:
            self.dismiss(None)
    
    def action_save(self) -> None:
        content = self.query_one("#edit-content", TextArea).text
        try:
            importance = float(self.query_one("#edit-importance", Input).value)
            importance = max(0.0, min(1.0, importance))
        except ValueError:
            importance = self.memory.importance
        
        tags_str = self.query_one("#edit-tags", Input).value
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        self.dismiss({
            "content": content,
            "importance": importance,
            "tags": tags
        })
    
    def action_cancel(self) -> None:
        self.dismiss(None)


class NewMemoryScreen(ModalScreen[dict]):
    """Modal screen for creating a new memory."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]
    
    def compose(self) -> ComposeResult:
        with Container(id="edit-dialog"):
            yield Label("Create New Memory", id="edit-title")
            
            yield Label("Content:")
            yield TextArea("", id="edit-content")
            
            yield Label("Importance (0.0 - 1.0):")
            yield Input("0.5", id="edit-importance")
            
            yield Label("Tags (comma-separated):")
            yield Input("", id="edit-tags")
            
            with Horizontal(id="edit-buttons"):
                yield Button("Create (Ctrl+S)", variant="success", id="save-btn")
                yield Button("Cancel (Esc)", variant="error", id="cancel-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self.action_save()
        else:
            self.dismiss(None)
    
    def action_save(self) -> None:
        content = self.query_one("#edit-content", TextArea).text
        if not content.strip():
            return  # Don't create empty memories
        
        try:
            importance = float(self.query_one("#edit-importance", Input).value)
            importance = max(0.0, min(1.0, importance))
        except ValueError:
            importance = 0.5
        
        tags_str = self.query_one("#edit-tags", Input).value
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        self.dismiss({
            "content": content,
            "importance": importance,
            "tags": tags
        })
    
    def action_cancel(self) -> None:
        self.dismiss(None)


class MemoryDetailPanel(Static):
    """Panel showing detailed information about a selected memory."""
    
    def __init__(self):
        super().__init__()
        self.memory = None
    
    def compose(self) -> ComposeResult:
        yield Static("Select a memory to view details", id="detail-content")
    
    def update_memory(self, memory) -> None:
        """Update the panel with a new memory's details."""
        self.memory = memory
        if memory is None:
            content = "Select a memory to view details"
        else:
            # Format age
            age_days = (datetime.now() - memory.timestamp).total_seconds() / 86400
            if age_days < 1:
                age_str = f"{int(age_days * 24)}h ago"
            elif age_days < 30:
                age_str = f"{int(age_days)}d ago"
            else:
                age_str = f"{int(age_days / 30)}mo ago"
            
            # Format last accessed
            if memory.last_accessed:
                last_acc_days = (datetime.now() - memory.last_accessed).total_seconds() / 86400
                if last_acc_days < 1:
                    last_acc_str = f"{int(last_acc_days * 24)}h ago"
                else:
                    last_acc_str = f"{int(last_acc_days)}d ago"
            else:
                last_acc_str = "Never"
            
            # Build content
            importance_stars = "★" * int(memory.importance * 5) + "☆" * (5 - int(memory.importance * 5))
            tags_str = ", ".join(memory.tags) if memory.tags else "(none)"
            archived_str = "🗑️ ARCHIVED" if getattr(memory, 'archived', False) else ""
            
            content = f"""[bold cyan]ID:[/] {memory.id}
{archived_str}

[bold cyan]Importance:[/] {importance_stars} ({memory.importance:.2f})

[bold cyan]Created:[/] {memory.timestamp.strftime('%Y-%m-%d %H:%M')} ({age_str})

[bold cyan]Last Accessed:[/] {last_acc_str}

[bold cyan]Access Count:[/] {memory.access_count}

[bold cyan]Tags:[/] {tags_str}

[bold cyan]Related Memories:[/] {len(memory.related_memories)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[bold cyan]Content:[/]
{memory.content}
"""
        
        self.query_one("#detail-content", Static).update(content)


class MemoryManager(App):
    """Interactive Memory Manager TUI."""
    
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 2fr 1fr;
    }
    
    #left-panel {
        height: 100%;
        border: solid $primary;
    }
    
    #right-panel {
        height: 100%;
        border: solid $secondary;
        padding: 1;
    }
    
    #memory-table {
        height: 1fr;
    }
    
    #search-container {
        height: auto;
        padding: 1;
    }
    
    #search-input {
        width: 100%;
    }
    
    #stats-bar {
        height: 3;
        padding: 0 1;
        background: $surface;
    }
    
    #detail-content {
        height: 100%;
    }
    
    MemoryDetailPanel {
        height: 100%;
        overflow-y: auto;
    }
    
    /* Dialog styling */
    #dialog {
        align: center middle;
        width: 50;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    
    #dialog-message {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }
    
    #dialog-buttons {
        width: 100%;
        align: center middle;
    }
    
    #dialog-buttons Button {
        margin: 0 1;
    }
    
    /* Edit dialog */
    #edit-dialog {
        width: 80;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    
    #edit-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    #edit-content {
        height: 10;
        margin-bottom: 1;
    }
    
    #edit-importance, #edit-tags {
        margin-bottom: 1;
    }
    
    #edit-buttons {
        margin-top: 1;
        align: center middle;
    }
    
    #edit-buttons Button {
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("a", "archive", "Archive"),
        Binding("u", "unarchive", "Unarchive"),
        Binding("e", "edit", "Edit"),
        Binding("n", "new", "New Memory"),
        Binding("t", "toggle_archived", "Toggle View"),
        Binding("r", "refresh", "Refresh"),
        Binding("/", "search", "Search"),
        Binding("escape", "clear_search", "Clear"),
        Binding("1", "sort_combined", "Sort: Combined"),
        Binding("2", "sort_importance", "Sort: Importance"),
        Binding("3", "sort_recency", "Sort: Recency"),
        Binding("4", "sort_access", "Sort: Access"),
    ]
    
    def __init__(self, memory_path: str = "vector_memory"):
        super().__init__()
        self.memory_path = Path(memory_path)
        self.metadata_file = self.memory_path / "metadata.pkl"
        self.memories = {}
        self.show_archived = False
        self.search_query = ""
        self.sort_mode = "combined"
        self.selected_memory = None
        self.vector_memory = None  # Will be loaded if available
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="left-panel"):
            with Container(id="search-container"):
                yield Input(placeholder="Search memories... (press / to focus)", id="search-input")
            yield Static("", id="stats-bar")
            yield DataTable(id="memory-table")
        
        with VerticalScroll(id="right-panel"):
            yield MemoryDetailPanel()
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.title = "🧠 Engram Memory Manager"
        self.sub_title = "Browse, Edit, Archive"
        
        # Set up data table
        table = self.query_one("#memory-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("★", "ID", "Age", "Acc", "Content")
        
        # Load and display memories
        self.load_memories()
        self.refresh_table()
    
    def load_memories(self) -> bool:
        """Load memories from the pickle file."""
        if not self.metadata_file.exists():
            self.memories = {}
            return False
        
        try:
            with open(self.metadata_file, 'rb') as f:
                metadata = CustomUnpickler(f).load()
                self.memories = metadata.get('memories', {})
                return True
        except Exception as e:
            self.memories = {}
            self.notify(f"Error loading memories: {e}", severity="error")
            return False
    
    def get_filtered_memories(self) -> List:
        """Get memories filtered by current view settings."""
        if not self.memories:
            return []
        
        memories = list(self.memories.values())
        
        # Filter by archived status - ensure archived attribute exists
        def is_archived(m):
            return getattr(m, 'archived', False) is True
        
        memories = [m for m in memories if is_archived(m) == self.show_archived]
        
        # Filter by search query
        if self.search_query:
            query_lower = self.search_query.lower()
            memories = [m for m in memories if 
                       query_lower in m.content.lower() or
                       any(query_lower in tag.lower() for tag in m.tags)]
        
        # Sort
        if self.sort_mode == "importance":
            memories.sort(key=lambda m: m.importance, reverse=True)
        elif self.sort_mode == "recency":
            memories.sort(key=lambda m: m.timestamp, reverse=True)
        elif self.sort_mode == "access":
            memories.sort(key=lambda m: m.access_count, reverse=True)
        else:  # combined
            max_access = max((m.access_count for m in memories), default=1) or 1
            now = datetime.now()
            def combined_score(m):
                age_days = (now - m.timestamp).total_seconds() / 86400
                recency = 0.5 ** (age_days / 30)
                return m.importance * 0.4 + recency * 0.3 + (m.access_count / max_access) * 0.3
            memories.sort(key=combined_score, reverse=True)
        
        return memories
    
    def refresh_table(self) -> None:
        """Refresh the memory table display."""
        table = self.query_one("#memory-table", DataTable)
        table.clear()
        
        memories = self.get_filtered_memories()
        
        if not memories:
            # Show friendly empty state message
            if self.show_archived:
                table.add_row("", "", "", "", "No archived memories... for now!")
            else:
                table.add_row("", "", "", "", "No memories... for now! Press 'n' to create one.")
        
        for mem in memories:
            # Importance stars (1-3)
            if mem.importance >= 0.7:
                stars = "★★★"
            elif mem.importance >= 0.4:
                stars = "★★☆"
            else:
                stars = "★☆☆"
            
            # Age
            age_days = (datetime.now() - mem.timestamp).total_seconds() / 86400
            if age_days < 1:
                age_str = f"{int(age_days * 24)}h"
            elif age_days < 30:
                age_str = f"{int(age_days)}d"
            else:
                age_str = f"{int(age_days / 30)}mo"
            
            # Truncated content
            content = mem.content.replace("\n", " ")[:50]
            if len(mem.content) > 50:
                content += "..."
            
            table.add_row(
                stars,
                mem.id[:8],
                age_str,
                str(mem.access_count),
                content,
                key=mem.id
            )
        
        # Update stats bar
        total = len(self.memories)
        active = len([m for m in self.memories.values() if not getattr(m, 'archived', False)])
        archived = total - active
        view_mode = "ARCHIVED" if self.show_archived else "ACTIVE"
        sort_label = self.sort_mode.upper()
        
        stats = f"📊 Total: {total} | Active: {active} | Archived: {archived} | View: {view_mode} | Sort: {sort_label}"
        if self.search_query:
            stats += f" | 🔍 '{self.search_query}'"
        
        self.query_one("#stats-bar", Static).update(stats)
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the table."""
        if event.row_key:
            memory_id = str(event.row_key.value)
            self.selected_memory = self.memories.get(memory_id)
            self.query_one(MemoryDetailPanel).update_memory(self.selected_memory)
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "search-input":
            self.search_query = event.value
            self.refresh_table()
    
    def action_search(self) -> None:
        """Focus the search input."""
        self.query_one("#search-input", Input).focus()
    
    def action_clear_search(self) -> None:
        """Clear search and unfocus."""
        search_input = self.query_one("#search-input", Input)
        search_input.value = ""
        self.search_query = ""
        self.refresh_table()
        self.query_one("#memory-table", DataTable).focus()
    
    def action_toggle_archived(self) -> None:
        """Toggle between active and archived view."""
        self.show_archived = not self.show_archived
        self.selected_memory = None
        self.query_one(MemoryDetailPanel).update_memory(None)
        self.refresh_table()
        view = "archived" if self.show_archived else "active"
        self.notify(f"Showing {view} memories")
    
    def action_refresh(self) -> None:
        """Reload memories from disk."""
        self.load_memories()
        self.refresh_table()
        self.notify("Memories refreshed")
    
    def action_sort_combined(self) -> None:
        self.sort_mode = "combined"
        self.refresh_table()
    
    def action_sort_importance(self) -> None:
        self.sort_mode = "importance"
        self.refresh_table()
    
    def action_sort_recency(self) -> None:
        self.sort_mode = "recency"
        self.refresh_table()
    
    def action_sort_access(self) -> None:
        self.sort_mode = "access"
        self.refresh_table()
    
    def _load_vector_memory(self):
        """Lazy-load VectorMemory for write operations."""
        if self.vector_memory is None:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from engram_pkg.core import VectorMemory
                self.vector_memory = VectorMemory(str(self.memory_path))
            except Exception as e:
                self.notify(f"Could not load VectorMemory: {e}", severity="error")
                return None
        return self.vector_memory
    
    def action_archive(self) -> None:
        """Archive the selected memory."""
        if not self.selected_memory:
            self.notify("No memory selected", severity="warning")
            return
        
        if self.show_archived:
            self.notify("Use 'u' to unarchive in archived view", severity="warning")
            return
        
        async def do_archive(confirmed: bool) -> None:
            if confirmed:
                vm = self._load_vector_memory()
                if vm:
                    success = vm.archive_memory(self.selected_memory.id)
                    if success:
                        vm._save_persistent_data()
                        self.load_memories()
                        self.selected_memory = None
                        self.query_one(MemoryDetailPanel).update_memory(None)
                        self.refresh_table()
                        self.notify("Memory archived")
                    else:
                        self.notify("Failed to archive memory", severity="error")
        
        content_preview = self.selected_memory.content[:50] + "..."
        self.push_screen(
            ConfirmDialog(f"Archive this memory?\n\n'{content_preview}'", "archive"),
            do_archive
        )
    
    def action_unarchive(self) -> None:
        """Unarchive the selected memory."""
        if not self.selected_memory:
            self.notify("No memory selected", severity="warning")
            return
        
        if not self.show_archived:
            self.notify("Memory is already active", severity="warning")
            return
        
        vm = self._load_vector_memory()
        if vm:
            success = vm.unarchive_memory(self.selected_memory.id)
            if success:
                vm._save_persistent_data()
                self.load_memories()
                self.selected_memory = None
                self.query_one(MemoryDetailPanel).update_memory(None)
                self.refresh_table()
                self.notify("Memory unarchived")
            else:
                self.notify("Failed to unarchive memory", severity="error")
    
    def action_edit(self) -> None:
        """Edit the selected memory."""
        if not self.selected_memory:
            self.notify("No memory selected", severity="warning")
            return
        
        async def handle_edit(result: Optional[dict]) -> None:
            if result:
                vm = self._load_vector_memory()
                if vm:
                    success = vm.update_memory(
                        self.selected_memory.id,
                        content=result["content"],
                        importance=result["importance"],
                        tags=result["tags"]
                    )
                    if success:
                        vm._save_persistent_data()
                        self.load_memories()
                        # Re-select the edited memory
                        self.selected_memory = self.memories.get(self.selected_memory.id)
                        self.query_one(MemoryDetailPanel).update_memory(self.selected_memory)
                        self.refresh_table()
                        self.notify("Memory updated")
                    else:
                        self.notify("Failed to update memory", severity="error")
        
        self.push_screen(EditMemoryScreen(self.selected_memory), handle_edit)
    
    def action_new(self) -> None:
        """Create a new memory."""
        async def handle_new(result: Optional[dict]) -> None:
            if result:
                vm = self._load_vector_memory()
                if vm:
                    memory_id = vm.store_memory(
                        content=result["content"],
                        importance=result["importance"],
                        tags=result["tags"]
                    )
                    vm._save_persistent_data()
                    self.load_memories()
                    self.refresh_table()
                    self.notify(f"Memory created: {memory_id[:8]}...")
        
        self.push_screen(NewMemoryScreen(), handle_new)


def main():
    parser = argparse.ArgumentParser(
        description="🧠 Interactive Memory Manager for Engram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keybindings:
  /          Search memories
  Escape     Clear search / Cancel
  n          Create new memory
  t          Toggle archived/active view
  a          Archive selected memory
  u          Unarchive selected memory (in archived view)
  e          Edit selected memory
  r          Refresh from disk
  1-4        Sort modes (combined/importance/recency/access)
  q          Quit
        """
    )
    
    parser.add_argument(
        "--memory-path",
        default="vector_memory",
        help="Path to vector memory storage (default: vector_memory)"
    )
    
    args = parser.parse_args()
    
    # Resolve path
    memory_path = args.memory_path
    if not Path(memory_path).is_absolute():
        script_dir = Path(__file__).parent
        memory_path = script_dir / memory_path
    
    app = MemoryManager(memory_path=str(memory_path))
    app.run()


if __name__ == "__main__":
    main()
