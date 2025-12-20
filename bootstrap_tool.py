#!/usr/bin/env python3
"""
🧠 MEMORY SYSTEM BOOTSTRAP TOOL
===============================

CRITICAL FIRST STEP for Cursor IDE conversations.
This tool MUST be called at the very beginning of every new conversation.

This provides the AI with essential knowledge about how to use the persistent
memory system, ensuring consistent self-awareness across conversations.

USAGE:
    python bootstrap_tool.py
    # or as MCP tool: bootstrap_memory_system()

OUTPUT:
    Essential memory system knowledge and usage instructions.
"""

import sys
import os
from pathlib import Path

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

try:
    from bootstrap_memory import execute_bootstrap_tool

    def main():
        """Main bootstrap tool execution"""
        print("=" * 80)
        print("🧠 MEMORY SYSTEM BOOTSTRAP TOOL")
        print("=" * 80)
        print()

        result = execute_bootstrap_tool()
        print(result)

        print()
        print("=" * 80)
        print("✅ BOOTSTRAP COMPLETE - Memory system ready!")
        print("=" * 80)

    def bootstrap_memory_system():
        """
        MCP Tool: Bootstrap memory system for new conversations.

        This tool provides essential knowledge about the persistent memory system
        and must be called at the beginning of every new conversation in Cursor IDE.

        Returns:
            Essential memory system knowledge and usage instructions.
        """
        return execute_bootstrap_tool()

    if __name__ == "__main__":
        main()

except ImportError as e:
    print("❌ CRITICAL ERROR: Bootstrap tool failed to load")
    print(f"Error: {e}")
    print()
    print("🔧 TROUBLESHOOTING:")
    print("1. Ensure you're in the project directory")
    print("2. Activate virtual environment: source .venv/bin/activate")
    print("3. Install dependencies: pip install -r requirements.txt")
    print()
    print("💡 MANUAL BOOTSTRAP:")
    print("If this tool fails, you can manually access bootstrap knowledge by")
    print("checking the README.md file or running: python quick_memory.py bootstrap")
    sys.exit(1)
