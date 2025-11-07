# Custom Airflow Image with Additional Tools
# Base: Official Apache Airflow 2.7.3
# Additions: unzip, postgresql-client-18, mdbtools

FROM apache/airflow:2.7.3

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    wget \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Add PostgreSQL 18 repository
RUN echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -

# Install PostgreSQL 18 client tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client-18 \
    && rm -rf /var/lib/apt/lists/*

# Install mdbtools for .mdb file access (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    mdbtools \
    && rm -rf /var/lib/apt/lists/*

# Verify installations
RUN unzip -v && pg_dump --version && mdb-ver -M

USER airflow

# Install Python dependencies (if any additional needed)
# COPY requirements.txt /tmp/
# RUN pip install --no-cache-dir -r /tmp/requirements.txt
