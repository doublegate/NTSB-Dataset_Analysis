# Contributing to NTSB Dataset Analysis

Thank you for your interest in contributing to the NTSB Aviation Accident Database Analysis project! This document provides guidelines for contributing to this repository.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful, inclusive, and considerate in all interactions.

### Our Standards

- Be welcoming and inclusive
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear descriptive title** for the issue
- **Detailed description** of the problem
- **Steps to reproduce** the behavior
- **Expected behavior** vs. actual behavior
- **Environment details**: OS, Python version, Fish shell version
- **Error messages** or logs (if applicable)
- **Screenshots** (if relevant)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When suggesting an enhancement:

- Use a **clear and descriptive title**
- Provide a **detailed description** of the proposed functionality
- Explain **why this enhancement would be useful**
- Include **examples** of how it would work
- Consider **implementation details** (if applicable)

### Contributing Code

We welcome contributions in several areas:

1. **Analysis Scripts**
   - Add new Python analysis examples
   - Create Jupyter notebooks demonstrating specific techniques
   - Develop visualization dashboards (Streamlit, Dash, Plotly)

2. **Data Processing Tools**
   - Improve extraction scripts (Fish shell)
   - Add data cleaning/preprocessing utilities
   - Optimize query performance
   - Add support for new data formats

3. **Documentation**
   - Improve existing documentation
   - Add tutorials and how-to guides
   - Create example use cases
   - Translate documentation (if multilingual support desired)

4. **Testing**
   - Add unit tests for Python scripts
   - Create integration tests for Fish shell scripts
   - Improve test coverage

5. **Infrastructure**
   - CI/CD improvements
   - Docker containerization
   - Package management enhancements

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- CachyOS/Arch Linux (or compatible distribution)
- Fish shell installed
- Python 3.11+ with venv
- Git for version control
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```fish
   git clone https://github.com/YOUR_USERNAME/NTSB-Dataset_Analysis.git
   cd NTSB-Dataset_Analysis
   ```

3. Add upstream remote:
   ```fish
   git remote add upstream https://github.com/ORIGINAL_OWNER/NTSB-Dataset_Analysis.git
   ```

### Setup Development Environment

```fish
# Run setup script
./setup.fish

# Activate Python virtual environment
source .venv/bin/activate.fish

# Extract sample data for testing
./scripts/extract_all_tables.fish datasets/avall.mdb
```

## Development Workflow

### Branching Strategy

- `main` - Stable production-ready code
- `develop` - Integration branch for features (if using Git Flow)
- `feature/*` - New features or enhancements
- `bugfix/*` - Bug fixes
- `docs/*` - Documentation updates
- `test/*` - Test additions or improvements

### Creating a Feature Branch

```fish
# Update your fork
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Making Changes

1. Make your changes in the feature branch
2. Test your changes thoroughly
3. Follow the coding standards (see below)
4. Update documentation as needed
5. Add or update tests if applicable

### Committing Changes

Use clear, descriptive commit messages following conventional commits format:

```
type(scope): brief description

Detailed explanation (if needed)

- Additional details
- Breaking changes (if any)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependencies, etc.

**Examples:**
```
feat(scripts): add parallel extraction for large databases

Implement parallel table extraction to improve performance when
processing multiple large MDB files simultaneously.

- Uses Fish shell background jobs
- Reduces extraction time by ~60% for avall.mdb
- Maintains compatibility with existing scripts
```

```
fix(analysis): correct fatal accident count calculation

The previous calculation included non-fatal injuries.
Now properly filters for inj_tot_f > 0 only.

Fixes #123
```

```
docs(readme): update installation instructions for macOS

Add specific instructions for Homebrew installation on macOS,
including mdbtools formula and Python setup.
```

## Coding Standards

### Python Code

- Follow **PEP 8** style guide
- Use **type hints** for function parameters and return values
- Write **docstrings** for all functions and classes (Google or NumPy style)
- Keep functions **focused and small** (single responsibility)
- Use **meaningful variable names**
- Add **comments** for complex logic

**Example:**
```python
def calculate_accident_rate(events: pd.DataFrame, year: int) -> float:
    """
    Calculate the accident rate for a specific year.

    Args:
        events: DataFrame containing event records with 'ev_year' column
        year: The year to calculate the rate for

    Returns:
        Accident rate as a float (accidents per 1000 flight hours)

    Raises:
        ValueError: If year is not present in the dataset
    """
    year_events = events[events['ev_year'] == year]
    if len(year_events) == 0:
        raise ValueError(f"No events found for year {year}")

    return len(year_events) / 1000.0  # Simplified calculation
```

### Fish Shell Scripts

- Use **clear variable names**
- Add **comments** explaining complex logic
- Include **usage examples** in script headers
- Handle **errors gracefully** with informative messages
- Test scripts on fresh environments

**Example:**
```fish
#!/usr/bin/env fish

# Extract a single table from an MDB database
# Usage: ./extract_table.fish <database.mdb> <table_name>
# Example: ./extract_table.fish datasets/avall.mdb events

# Check arguments
if test (count $argv) -ne 2
    echo "Error: Requires exactly 2 arguments"
    echo "Usage: $argv[0] <database.mdb> <table_name>"
    exit 1
end

set db_file $argv[1]
set table_name $argv[2]

# Validate database file exists
if not test -f $db_file
    echo "Error: Database file not found: $db_file"
    exit 1
end

# Extract table
mdb-export $db_file $table_name > "data/$table_name.csv"
```

### Documentation

- Use **Markdown** for all documentation
- Include **code examples** where appropriate
- Keep documentation **up-to-date** with code changes
- Use **clear headings** and **table of contents** for long documents
- Add **links** to related documentation

## Testing Guidelines

### Python Tests

Use `pytest` for Python testing:

```fish
# Install pytest
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Fish Shell Tests

Test Fish scripts manually:

```fish
# Test extraction script
./scripts/extract_table.fish datasets/avall.mdb events

# Verify output
ls -lh data/events.csv
head -n 5 data/events.csv
```

### Test Data

- Use **small sample datasets** for testing when possible
- Do **not commit large test data** to the repository
- Document **test data requirements** clearly

## Documentation

### What to Document

1. **Code Comments**
   - Explain **why**, not what (code shows what)
   - Document **non-obvious** logic
   - Add **TODO** or **FIXME** comments for future work

2. **Docstrings**
   - All public functions, classes, and modules
   - Parameters, return values, exceptions
   - Usage examples

3. **README Updates**
   - New features or capabilities
   - Changed installation requirements
   - Updated usage examples

4. **CHANGELOG**
   - Add entry to `CHANGELOG.md` under `[Unreleased]`
   - Follow Keep a Changelog format
   - Include breaking changes prominently

## Submitting Changes

### Pull Request Process

1. **Update Documentation**
   - Update README.md if needed
   - Add entry to CHANGELOG.md
   - Update relevant documentation files

2. **Test Your Changes**
   - Run all tests
   - Test manually in a clean environment
   - Verify no regressions

3. **Create Pull Request**
   - Push to your fork
   - Open PR against `main` branch (or `develop` if using Git Flow)
   - Use descriptive PR title and description
   - Link related issues with "Fixes #123" or "Closes #456"

4. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix (non-breaking change fixing an issue)
   - [ ] New feature (non-breaking change adding functionality)
   - [ ] Breaking change (fix or feature causing existing functionality to change)
   - [ ] Documentation update

   ## Testing
   - [ ] Tested locally
   - [ ] Added/updated tests
   - [ ] All tests pass

   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] No new warnings generated

   ## Related Issues
   Fixes #123
   ```

5. **Code Review**
   - Respond to review comments
   - Make requested changes
   - Re-request review after updates

6. **Merge**
   - Once approved, a maintainer will merge your PR
   - Delete your feature branch after merge

### Review Process

Maintainers will review your PR for:

- **Code quality** and adherence to standards
- **Test coverage** and passing tests
- **Documentation** completeness
- **Compatibility** with existing features
- **Performance** considerations

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### Recognition

Contributors will be:
- Listed in repository contributors
- Mentioned in CHANGELOG.md
- Credited in release notes

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search closed issues and PRs
3. Open a new issue with the "question" label
4. Reach out to maintainers

Thank you for contributing to NTSB Dataset Analysis!
