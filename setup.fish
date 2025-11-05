#!/usr/bin/env fish
# NTSB Database Analysis Environment Setup
# For CachyOS Linux with Fish shell

echo "🔧 Setting up NTSB Database Analysis Environment..."

# System packages
echo "📦 Installing system packages..."
sudo pacman -S --needed sqlite python python-pip \
    gdal hdf5 jq yq bat ripgrep fd fzf postgresql cmake base-devel \
    gettext autoconf-archive txt2man

# AUR packages (requires paru)
if command -v paru > /dev/null 2>&1
    echo "📦 Installing AUR packages..."
    paru -S --needed mdbtools duckdb dbeaver quarto-cli
else
    echo "⚠️  paru not found. Skipping AUR packages (mdbtools, duckdb, dbeaver, quarto-cli)"
    echo "   Install paru: https://github.com/morganamilo/paru"
    echo "   Note: mdbtools and duckdb are required and only available via AUR"
end

# Python virtual environment (recommended)
echo "🐍 Setting up Python virtual environment..."
if not test -d .venv
    python -m venv .venv
    echo "✅ Created virtual environment at .venv"
end

echo "📦 Installing Python packages..."
source .venv/bin/activate.fish

# Core data science
pip install --upgrade pip wheel setuptools
pip install pandas polars numpy scipy statsmodels scikit-learn

# Visualization
pip install matplotlib seaborn plotly altair

# Geospatial
pip install geopandas folium geopy shapely

# Text analysis
pip install nltk spacy wordcloud textblob

# Jupyter & notebooks
pip install jupyter jupyterlab ipython jupyterlab-git

# Dashboards
pip install streamlit dash panel

# High performance
pip install dask[complete] pyarrow fastparquet

# Database tools
pip install duckdb sqlalchemy

# CLI tools
pip install csvkit

# Utilities
pip install python-dateutil pytz

echo "📦 Downloading NLP models..."
python -m spacy download en_core_web_sm

# Rust tools (if cargo is available)
if command -v cargo > /dev/null 2>&1
    echo "🦀 Installing Rust tools..."

    # Install xsv (simpler CSV tool)
    echo "  → Installing xsv..."
    if cargo install xsv --locked > /dev/null 2>&1
        echo "     ✅ xsv installed successfully"
    else
        echo "     ⚠️  xsv installation failed (may already be installed)"
    end

    # Install qsv with full features (may fail on current version)
    echo "  → Checking qsv installation..."
    if command -v qsv > /dev/null 2>&1
        set qsv_version (qsv --version 2>&1 | head -n1)
        echo "     ℹ️  qsv already installed: $qsv_version"
        echo "     Skipping installation"
    else
        echo "     Attempting to install qsv (with full features)..."
        echo "     Note: qsv v9.1.0 has known compilation issues"
        if cargo install qsv --locked --features feature_capable > /dev/null 2>&1
            echo "     ✅ qsv installed successfully"
        else
            echo "     ⚠️  qsv installation failed - this is a known issue with v9.1.0"
            echo "     Alternative: Use xsv (already installed) or wait for qsv v9.1.1"
            echo "     You can manually install from git: cargo install --git https://github.com/jqnatividad/qsv qsv --features='feature_capable'"
        end
    end

    # Install polars-cli (optional - may take a while)
    echo "  → Installing polars-cli (this may take several minutes)..."
    if cargo install polars-cli --locked > /dev/null 2>&1
        echo "     ✅ polars-cli installed successfully"
    else
        echo "     ⚠️  polars-cli installation failed (may already be installed)"
    end

    # Install datafusion-cli (optional)
    echo "  → Installing datafusion-cli..."
    if cargo install datafusion-cli --locked > /dev/null 2>&1
        echo "     ✅ datafusion-cli installed successfully"
    else
        echo "     ⚠️  datafusion-cli installation failed (may already be installed)"
    end

    echo "✅ Rust tools installation complete"
else
    echo "⚠️  cargo not found. Skipping Rust tools"
    echo "   Install Rust first: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
end

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data examples scripts outputs figures

# Verify helper scripts exist
echo "✅ Verifying helper scripts..."
if test -f scripts/extract_all_tables.fish
    chmod +x scripts/extract_all_tables.fish
    echo "  ✓ scripts/extract_all_tables.fish"
else
    echo "  ⚠️  scripts/extract_all_tables.fish not found"
end

if test -f examples/quick_analysis.py
    chmod +x examples/quick_analysis.py
    echo "  ✓ examples/quick_analysis.py"
else
    echo "  ⚠️  examples/quick_analysis.py not found"
end

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo "  1. Extract data: ./scripts/extract_all_tables.fish datasets/avall.mdb"
echo "  2. Activate venv: source .venv/bin/activate.fish"
echo "  3. Start Jupyter: jupyter lab"
echo "  4. Run example: python examples/quick_analysis.py"
echo ""
echo "📖 See TOOLS_AND_UTILITIES.md for detailed documentation"
