# NTSB Aviation Accident Database - Production Dockerfile
# Phase 3 Sprint 1: PostgreSQL 18 with PostGIS extension

FROM postgres:18-alpine

LABEL maintainer="NTSB Aviation Database Project"
LABEL description="PostgreSQL 18 with PostGIS for NTSB aviation accident data"
LABEL version="18.0"

# Install PostGIS and other extensions
RUN apk add --no-cache \
    postgis \
    postgis-utils \
    && rm -rf /var/cache/apk/*

# Copy custom PostgreSQL configuration
# Optimized for analytics workload (read-heavy, complex queries)
COPY postgresql.conf /etc/postgresql/postgresql.conf

# Copy initialization scripts
# These run automatically on first container startup
COPY init/*.sql /docker-entrypoint-initdb.d/

# Environment variables (can be overridden in docker-compose.yml)
ENV POSTGRES_DB=ntsb_aviation \
    POSTGRES_USER=ntsb_user \
    PGDATA=/var/lib/postgresql/data/pgdata

# Expose PostgreSQL port
EXPOSE 5432

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} || exit 1

# Use default postgres entrypoint
# This handles initialization, configuration, and startup
