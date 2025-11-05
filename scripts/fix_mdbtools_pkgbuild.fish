#!/usr/bin/env fish
# Fix mdbtools PKGBUILD to work with current gettext
# Based on: https://github.com/mdbtools/mdbtools/issues/370

echo "🔧 Fixing mdbtools PKGBUILD..."

# Create temporary directory for the fix
set temp_dir (mktemp -d)
cd $temp_dir

# Clone the AUR package
echo "📦 Cloning mdbtools from AUR..."
git clone https://aur.archlinux.org/mdbtools.git
cd mdbtools

# Backup original PKGBUILD
cp PKGBUILD PKGBUILD.orig

# Apply the fix: add -I /usr/share/gettext/m4 to autoreconf
echo "✏️  Patching PKGBUILD..."
sed -i 's/autoreconf -i -f$/autoreconf -i -f -I \/usr\/share\/gettext\/m4/' PKGBUILD

# Show the diff
echo ""
echo "📋 Changes made:"
diff -u PKGBUILD.orig PKGBUILD || true
echo ""

# Build and install
echo "🔨 Building mdbtools with patched PKGBUILD..."
makepkg -si

# Cleanup
cd
rm -rf $temp_dir

echo ""
echo "✅ mdbtools installation complete!"
