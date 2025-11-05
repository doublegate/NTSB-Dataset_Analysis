# mdbtools Installation Fix

## 🔴 Problem

The mdbtools package from AUR fails to build with the following error:

```
configure:21182: error: possibly undefined macro: AC_LIB_PREPARE_PREFIX
configure:21183: error: possibly undefined macro: AC_LIB_RPATH
configure:21188: error: possibly undefined macro: AC_LIB_LINKFLAGS_BODY
configure:21196: error: possibly undefined macro: AC_LIB_APPENDTOVAR
autoreconf: error: /usr/bin/autoconf failed with exit status: 1
==> ERROR: A failure occurred in prepare().
```

## 🔍 Root Cause

The mdbtools PKGBUILD uses `autoreconf -i -f` but doesn't tell autoreconf where to find the gettext m4 macros. Even though gettext, autoconf-archive, and txt2man are installed, autoreconf can't locate the required macros without the explicit path.

**Upstream Issue:** https://github.com/mdbtools/mdbtools/issues/370

## ✅ Solution

The fix is to modify the PKGBUILD to add `-I /usr/share/gettext/m4` to the autoreconf command.

### Option 1: Automated Fix (Recommended)

Use the provided script:

```fish
./fix_mdbtools_pkgbuild.fish
```

This script will:
1. Clone mdbtools from AUR
2. Patch the PKGBUILD automatically
3. Build and install the package

### Option 2: Manual Fix

1. Clone the package:
   ```fish
   cd /tmp
   git clone https://aur.archlinux.org/mdbtools.git
   cd mdbtools
   ```

2. Edit the PKGBUILD file:
   ```fish
   vim PKGBUILD  # or use your preferred editor
   ```

3. Find the `prepare()` function and change:
   ```bash
   prepare() {
     cd "${srcdir}/${_srcname}"
     autoreconf -i -f
   }
   ```

   To:
   ```bash
   prepare() {
     cd "${srcdir}/${_srcname}"
     autoreconf -i -f -I /usr/share/gettext/m4
   }
   ```

4. Build and install:
   ```fish
   makepkg -si
   ```

## 🔧 Alternative: Use sed to patch automatically

```fish
cd /tmp
git clone https://aur.archlinux.org/mdbtools.git
cd mdbtools
sed -i 's/autoreconf -i -f$/autoreconf -i -f -I \/usr\/share\/gettext\/m4/' PKGBUILD
makepkg -si
```

## 📝 What This Fix Does

The `-I /usr/share/gettext/m4` flag tells autoreconf to look for m4 macro files in `/usr/share/gettext/m4`, which is where gettext installs its autoconf macros:

- `AC_LIB_PREPARE_PREFIX`
- `AC_LIB_RPATH`
- `AC_LIB_LINKFLAGS_BODY`
- `AC_LIB_APPENDTOVAR`

Without this explicit path, autoreconf searches only in the default locations and fails to find these macros.

## 🎯 Verification

After installation, verify mdbtools works:

```fish
# Check installation
mdb-ver --version

# Test with a database
mdb-tables datasets/avall.mdb

# Export a table
mdb-export datasets/avall.mdb events > /tmp/events.csv
```

## 📚 Related Documentation

- **Full Installation Guide:** See `INSTALLATION.md`
- **Quick Reference:** See `QUICKSTART.md`
- **GitHub Issue:** https://github.com/mdbtools/mdbtools/issues/370
- **AUR Package:** https://aur.archlinux.org/packages/mdbtools

## 🤝 Future Status

This issue should eventually be fixed in the AUR PKGBUILD itself. Check the AUR package comments for updates. Until then, use the provided fix script or manual patch method.
