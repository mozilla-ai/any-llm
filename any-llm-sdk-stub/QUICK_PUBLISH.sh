#!/bin/bash
# Quick publishing script for any-llm-sdk stub package
# Run this from the any-llm-sdk-stub directory

set -e  # Exit on error

echo "🚀 Publishing any-llm-sdk stub package to PyPI"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this from the any-llm-sdk-stub directory."
    exit 1
fi

# Check if any-llm is published
echo "🔍 Checking if any-llm is available on PyPI..."
if pip index versions any-llm 2>&1 | grep -q "No matching distribution"; then
    echo "❌ Error: 'any-llm' package not found on PyPI."
    echo "   You MUST publish the main 'any-llm' package first!"
    exit 1
fi
echo "✅ Main package 'any-llm' found on PyPI"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
echo ""

# Build the package
echo "📦 Building package..."
python -m build
echo ""

# Check the package
echo "🔍 Checking package..."
twine check dist/*
echo ""

# Ask for confirmation
read -p "Ready to upload to PyPI? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Upload to PyPI
echo "📤 Uploading to PyPI..."
twine upload dist/*
echo ""

echo "✅ Successfully published to PyPI!"
echo ""
echo "🧪 Test installation with:"
echo "   pip install any-llm-sdk"
echo ""
