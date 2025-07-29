#!/bin/bash

# Helper script to identify files larger than 100MB in the repository
# Useful for understanding what files might be blocked by the pre-commit hook

# Maximum file size in bytes (100MB = 100 * 1024 * 1024)
MAX_SIZE=104857600

echo "🔍 Scanning repository for files larger than 100MB..."
echo "=================================================="

large_files_found=false

# Find all files (excluding .git directory) and check their size
while IFS= read -r -d '' file; do
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        size=$(stat -f%z "$file" 2>/dev/null)
    else
        # Linux
        size=$(stat -c%s "$file" 2>/dev/null)
    fi
    
    if [ -n "$size" ] && [ "$size" -gt "$MAX_SIZE" ]; then
        # Convert size to MB for display
        size_mb=$((size / 1024 / 1024))
        echo "📁 $file (${size_mb}MB)"
        large_files_found=true
    fi
done < <(find . -type f -not -path './.git/*' -print0)

if [ "$large_files_found" = false ]; then
    echo "✅ No files larger than 100MB found!"
else
    echo ""
    echo "📋 Summary:"
    echo "These files are larger than 100MB and would be blocked by the pre-commit hook."
    echo "Consider adding them to .gitignore if they shouldn't be version controlled."
fi

echo ""
echo "- Add large data files to .gitignore"
echo "- Use 'git rm --cached <file>' to remove large files from staging"
echo "- Check .gitignore to see what's already excluded" 