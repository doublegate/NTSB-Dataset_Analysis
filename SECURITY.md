# Security Policy

## Supported Versions

The following versions of the NTSB Dataset Analysis project are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of the NTSB Dataset Analysis project seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Report vulnerabilities via:
   - GitHub Security Advisories (preferred): https://github.com/doublegate/NTSB-Dataset_Analysis/security/advisories
   - Or email the maintainers directly with details

### What to Include

When reporting a vulnerability, please include:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact and severity
- Any proof-of-concept code (if applicable)
- Suggested fix (if you have one)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Development**: Varies by severity and complexity
- **Public Disclosure**: Coordinated with reporter after fix is deployed

### Security Considerations for This Project

#### Data Security

This project works with **public domain data** from the NTSB. The aviation accident data is:

- **Public information** available from the NTSB website
- **No personal identifiable information (PII)** included
- **No confidential or sensitive data**

#### Code Security

Security considerations for the codebase:

1. **Dependencies**: Regularly update Python packages and Rust tools
2. **File Operations**: Scripts work with local files only, no external network operations
3. **Database Access**: MDB files are read-only operations via mdbtools
4. **No Authentication**: This is a data analysis toolkit, not a web service

#### Known Security Practices

- All scripts are open-source and auditable
- No external data transmission or API calls
- No credential storage or authentication mechanisms
- Local-only file operations within project directory

## Security Best Practices for Users

When using this project:

1. **Verify Script Sources**: Review Fish shell scripts before execution
2. **Virtual Environment**: Always use Python virtual environment (.venv/)
3. **File Permissions**: Ensure proper file permissions for datasets/
4. **Updates**: Keep dependencies up to date with `pip install --upgrade`
5. **Audit Tools**: Review third-party tools (mdbtools, DuckDB, etc.) before installation

## Scope

### In Scope

- Vulnerabilities in project scripts (Python, Fish shell)
- Issues with documentation that could lead to insecure usage
- Dependency vulnerabilities in requirements.txt

### Out of Scope

- NTSB data quality or accuracy issues (report to NTSB directly)
- Third-party tools (mdbtools, DuckDB, pandas, etc.)
- Operating system or Python interpreter vulnerabilities
- Issues requiring physical access to the user's machine

## Security Tools

Recommended security scanning tools for contributors:

- **Python**: `bandit`, `safety`, `pip-audit`
- **Dependencies**: `pip-audit`, Dependabot (GitHub)
- **Code Review**: Manual review of all scripts

## Responsible Disclosure

We follow a responsible disclosure policy:

1. Report is received and acknowledged
2. Vulnerability is verified and assessed
3. Fix is developed and tested
4. Security advisory is published (if warranted)
5. Public disclosure after fix is deployed

## Contact

For security-related questions or concerns:

- GitHub: Open an issue with the "security" label (for non-sensitive questions)
- Security Advisories: https://github.com/doublegate/NTSB-Dataset_Analysis/security/advisories

Thank you for helping keep the NTSB Dataset Analysis project secure!
