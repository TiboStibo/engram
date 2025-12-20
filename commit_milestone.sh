#!/bin/bash
# Script to commit significant self-improvement milestones
# Usage: ./commit_milestone.sh "Milestone description"

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"Milestone description\""
    echo "Example: $0 \"Implemented advanced memory consolidation algorithm\""
    exit 1
fi

cd /home/rob/Dev/persistent_memory_project

# Add all changes
git add -A

# Create milestone commit
git commit -m "🎯 Milestone: $1

- Significant self-improvement achievement
- $(date)
- Part of ongoing AI memory system development"

echo "✅ Milestone committed: $1"
echo "Commit details:"
git log --oneline -1
