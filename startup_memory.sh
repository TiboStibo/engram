#!/bin/bash
# AI Memory System Startup Bootstrap
# Run this at the beginning of any new conversation to initialize memory capabilities

echo "🧠 AI MEMORY SYSTEM STARTUP"
echo "=========================="
echo ""

# Check if we're in the right directory
if [ ! -f "bootstrap_memory.py" ]; then
    echo "❌ Error: bootstrap_memory.py not found. Please run from the memory project directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies if needed
echo "📦 Checking dependencies..."
pip install --quiet -r requirements.txt

# Bootstrap the memory system
echo "🧠 Bootstrapping memory system..."
python bootstrap_memory.py

# Show quick usage instructions
echo ""
echo "🎯 MEMORY SYSTEM READY!"
echo "======================="
echo ""
echo "Quick commands for this conversation:"
echo "  Query: python quick_memory.py query \"topic\""
echo "  Add:   python quick_memory.py add \"content\" \"tags\" 0.8"
echo "  Live:  python conversation_memory_assistant.py --live"
echo "  Stats: python quick_memory.py stats"
echo ""
echo "💡 Always query memories before responding to complex questions!"
echo ""

# Keep terminal open for interaction
echo "💻 Memory system is active. You can now use memory commands above."
echo "   Press Ctrl+C to exit when done."
echo ""

# Optional: Start live assistant if requested
if [ "$1" = "--live" ]; then
    echo "🎯 Starting live conversation assistant..."
    python conversation_memory_assistant.py --live
fi
