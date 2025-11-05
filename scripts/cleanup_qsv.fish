#!/usr/bin/env fish
# Cleanup failed qsv v9.1.0 installation

echo "🧹 Cleaning up failed qsv installation..."

# 1. Remove qsv binary if partially installed
echo "📦 Removing qsv binary (if exists)..."
if cargo uninstall qsv > /dev/null 2>&1
    echo "   ✅ Removed qsv binary"
else
    echo "   ℹ️  No qsv binary found (expected for failed install)"
end

# 2. Clean up intermediate build artifacts (usually auto-cleaned, but check)
echo "🗑️  Cleaning build artifacts..."
set tmp_dirs (find /tmp -maxdepth 1 -name "cargo-install*" -type d 2>/dev/null)
if test (count $tmp_dirs) -gt 0
    for dir in $tmp_dirs
        echo "   → Removing $dir"
        rm -rf $dir
    end
    echo "   ✅ Removed temporary build directories"
else
    echo "   ℹ️  No temporary build directories found"
end

# 3. Clean cargo registry cache (optional - removes qsv source cache)
echo "🗂️  Cleaning cargo registry cache for qsv..."
set cache_dir ~/.cargo/registry/cache
set src_dir ~/.cargo/registry/src
if test -d $cache_dir
    find $cache_dir -name "qsv-*.crate" -delete 2>/dev/null
    echo "   ✅ Cleaned qsv .crate files from cache"
end
if test -d $src_dir
    find $src_dir -type d -name "qsv-*" -exec rm -rf {} + 2>/dev/null
    echo "   ✅ Cleaned qsv source directories"
end

# 4. Optional: Clean ALL unused cargo cache (more aggressive)
echo ""
echo "📊 Current cargo cache size:"
if command -v du > /dev/null 2>&1
    du -sh ~/.cargo/registry 2>/dev/null || echo "   Unable to calculate"
end

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "💡 Optional: For deeper cleanup of ALL unused dependencies:"
echo "   cargo install cargo-cache"
echo "   cargo cache --autoclean"
echo ""
echo "   This will remove dependencies not used by any installed tools."
