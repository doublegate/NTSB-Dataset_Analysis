# Backup Database - Automated Database Backup

Create timestamped PostgreSQL database backups with compression, verification, and retention management.

---

## OBJECTIVE

Automated backup solution that:
- Creates pg_dump backups (full, schema-only, or data-only)
- Compresses with gzip for space efficiency
- Stores in organized backups/ directory
- Verifies backup integrity
- Reports backup size and location
- Supports retention policy (keep N most recent)
- Safe for regular use (read-only, no locking)

**Time Estimate:** 2-5 minutes (depending on database size)
**Output:** Compressed SQL dump file

---

## CONTEXT

**Project:** NTSB Aviation Database (PostgreSQL data repository)
**Repository:** /home/parobek/Code/NTSB_Datasets
**Database:** ntsb_aviation (current size: ~966 MB)
**Backup Directory:** backups/

**Backup Types:**
- **full** - Complete database (schema + data) - Default
- **schema** - Schema only (tables, indexes, functions)
- **data** - Data only (no schema)

---

## USAGE

```bash
/backup-db                       # Full backup (default)
/backup-db full                  # Full backup (explicit)
/backup-db schema                # Schema only
/backup-db data                  # Data only
/backup-db --name my_backup      # Custom name prefix
/backup-db --keep 10             # Keep only 10 most recent backups
/backup-db --verify              # Verify backup after creation
```

---

## EXECUTION PHASES

### PHASE 1: ENVIRONMENT CHECKS (1 minute)

```bash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💾 DATABASE BACKUP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check PostgreSQL tools
echo "Checking PostgreSQL tools..."
if ! command -v pg_dump &> /dev/null; then
    echo "❌ ERROR: pg_dump not found"
    echo "   Install: sudo pacman -S postgresql (or apt/brew equivalent)"
    exit 1
fi
echo "✅ pg_dump found: $(which pg_dump)"

# Check gzip
if ! command -v gzip &> /dev/null; then
    echo "❌ ERROR: gzip not found (required for compression)"
    exit 1
fi
echo "✅ gzip found"

# Check database connection
echo "Checking database connection..."
if ! psql -d ntsb_aviation -c "SELECT 1;" &> /dev/null; then
    echo "❌ ERROR: Cannot connect to ntsb_aviation database"
    exit 1
fi
echo "✅ Database connection verified"
echo ""

# Create backups directory
mkdir -p backups/
echo "✅ Backup directory: $(pwd)/backups/"
echo ""
```

---

### PHASE 2: PARSE ARGUMENTS (1 minute)

```bash
# Default values
BACKUP_TYPE="full"
CUSTOM_NAME=""
KEEP_COUNT=""
VERIFY_BACKUP=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        full|schema|data)
            BACKUP_TYPE="$1"
            ;;
        --name)
            CUSTOM_NAME="$2"
            shift
            ;;
        --keep)
            KEEP_COUNT="$2"
            shift
            ;;
        --verify)
            VERIFY_BACKUP="true"
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Generate backup filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "$CUSTOM_NAME" ]; then
    BACKUP_FILE="backups/${CUSTOM_NAME}_${BACKUP_TYPE}_${TIMESTAMP}.sql.gz"
else
    BACKUP_FILE="backups/ntsb_aviation_${BACKUP_TYPE}_${TIMESTAMP}.sql.gz"
fi

echo "Backup configuration:"
echo "  Type: $BACKUP_TYPE"
echo "  File: $BACKUP_FILE"
[ -n "$KEEP_COUNT" ] && echo "  Retention: Keep $KEEP_COUNT most recent"
[ "$VERIFY_BACKUP" = "true" ] && echo "  Verification: Enabled"
echo ""
```

---

### PHASE 3: PRE-BACKUP STATE (1 minute)

```bash
echo "📊 Pre-backup database state..."

# Database size
DB_SIZE=$(psql -d ntsb_aviation -t -c "SELECT pg_size_pretty(pg_database_size('ntsb_aviation'));" | xargs)
echo "  Database size: $DB_SIZE"

# Row counts
TOTAL_ROWS=$(psql -d ntsb_aviation -t -c "
    SELECT SUM(n_live_tup) 
    FROM pg_stat_user_tables 
    WHERE schemaname = 'public';
" | xargs)
echo "  Total rows: $TOTAL_ROWS"

# Table count
TABLE_COUNT=$(psql -d ntsb_aviation -t -c "
    SELECT COUNT(*) 
    FROM pg_tables 
    WHERE schemaname = 'public';
" | xargs)
echo "  Tables: $TABLE_COUNT"

echo ""
```

---

### PHASE 4: CREATE BACKUP (2-4 minutes)

```bash
echo "💾 Creating backup..."
echo "  This may take 2-5 minutes for large databases..."
echo ""

START_TIME=$(date +%s)

# Build pg_dump command based on backup type
case $BACKUP_TYPE in
    full)
        echo "  Backing up: Full database (schema + data)"
        pg_dump -d ntsb_aviation \
            --verbose \
            --format=plain \
            --no-owner \
            --no-privileges \
            2>&1 | gzip > "$BACKUP_FILE"
        DUMP_EXIT_CODE=${PIPESTATUS[0]}
        ;;
    schema)
        echo "  Backing up: Schema only (no data)"
        pg_dump -d ntsb_aviation \
            --verbose \
            --format=plain \
            --schema-only \
            --no-owner \
            --no-privileges \
            2>&1 | gzip > "$BACKUP_FILE"
        DUMP_EXIT_CODE=${PIPESTATUS[0]}
        ;;
    data)
        echo "  Backing up: Data only (no schema)"
        pg_dump -d ntsb_aviation \
            --verbose \
            --format=plain \
            --data-only \
            --no-owner \
            --no-privileges \
            2>&1 | gzip > "$BACKUP_FILE"
        DUMP_EXIT_CODE=${PIPESTATUS[0]}
        ;;
    *)
        echo "❌ ERROR: Invalid backup type: $BACKUP_TYPE"
        exit 1
        ;;
esac

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $DUMP_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Backup failed with exit code $DUMP_EXIT_CODE"
    echo "   Check database connection and disk space"
    exit $DUMP_EXIT_CODE
fi

echo ""
echo "✅ Backup created successfully in ${DURATION}s"
echo ""
```

---

### PHASE 5: VERIFY BACKUP (Optional, 1 minute)

```bash
if [ "$VERIFY_BACKUP" = "true" ]; then
    echo "🔍 Verifying backup integrity..."
    
    # Test gzip integrity
    if gzip -t "$BACKUP_FILE" 2>/dev/null; then
        echo "  ✅ Gzip integrity: OK"
    else
        echo "  ❌ ERROR: Backup file corrupted (gzip test failed)"
        exit 1
    fi
    
    # Check if file contains SQL commands
    if zcat "$BACKUP_FILE" 2>/dev/null | head -50 | grep -q "PostgreSQL database dump"; then
        echo "  ✅ Content validity: OK (PostgreSQL dump detected)"
    else
        echo "  ⚠️  WARNING: File may not be a valid PostgreSQL dump"
    fi
    
    # Count lines (rough data completeness check)
    LINE_COUNT=$(zcat "$BACKUP_FILE" 2>/dev/null | wc -l)
    echo "  ✅ Line count: $LINE_COUNT lines"
    
    if [ $LINE_COUNT -lt 100 ]; then
        echo "  ⚠️  WARNING: Backup seems unusually small"
    fi
    
    echo "  ✅ Backup verification passed"
    echo ""
fi
```

---

### PHASE 6: BACKUP STATISTICS (1 minute)

```bash
echo "📊 Backup statistics..."

# File size
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
BACKUP_SIZE_BYTES=$(du -b "$BACKUP_FILE" | cut -f1)
echo "  Compressed size: $BACKUP_SIZE"

# Uncompressed size estimate
UNCOMPRESSED_SIZE=$(zcat "$BACKUP_FILE" 2>/dev/null | wc -c | numfmt --to=iec-i --suffix=B)
echo "  Uncompressed size (est): $UNCOMPRESSED_SIZE"

# Compression ratio
COMPRESSION_RATIO=$(echo "scale=1; $BACKUP_SIZE_BYTES * 100 / $(zcat "$BACKUP_FILE" | wc -c)" | bc 2>/dev/null || echo "N/A")
if [ "$COMPRESSION_RATIO" != "N/A" ]; then
    echo "  Compression ratio: ${COMPRESSION_RATIO}% ($(echo "scale=1; 100 - $COMPRESSION_RATIO" | bc)% reduction)"
fi

# MD5 checksum
if command -v md5sum &> /dev/null; then
    CHECKSUM=$(md5sum "$BACKUP_FILE" | cut -d' ' -f1)
    echo "  MD5 checksum: $CHECKSUM"
fi

# Duration
echo "  Duration: ${DURATION}s"
echo "  Throughput: $(echo "scale=1; $BACKUP_SIZE_BYTES / $DURATION / 1024 / 1024" | bc 2>/dev/null || echo "N/A") MB/s"
echo ""
```

---

### PHASE 7: RETENTION MANAGEMENT (Optional, 1 minute)

```bash
if [ -n "$KEEP_COUNT" ]; then
    echo "🗑️  Managing backup retention..."
    
    # Count existing backups of same type
    BACKUP_COUNT=$(ls -1 backups/*_${BACKUP_TYPE}_*.sql.gz 2>/dev/null | wc -l)
    echo "  Current ${BACKUP_TYPE} backups: $BACKUP_COUNT"
    
    if [ "$BACKUP_COUNT" -gt "$KEEP_COUNT" ]; then
        # Calculate how many to delete
        DELETE_COUNT=$((BACKUP_COUNT - KEEP_COUNT))
        echo "  Deleting $DELETE_COUNT old backups (keeping $KEEP_COUNT most recent)..."
        
        # Delete oldest backups
        ls -1t backups/*_${BACKUP_TYPE}_*.sql.gz 2>/dev/null | tail -n +$((KEEP_COUNT + 1)) | while read old_backup; do
            echo "    Removing: $(basename $old_backup)"
            rm -f "$old_backup"
        done
        
        echo "  ✅ Retention policy applied"
    else
        echo "  ✅ No cleanup needed (under retention limit)"
    fi
    echo ""
fi
```

---

### PHASE 8: GENERATE BACKUP METADATA (1 minute)

```bash
echo "📄 Generating backup metadata..."

METADATA_FILE="${BACKUP_FILE%.sql.gz}.metadata.txt"

cat > "$METADATA_FILE" << EOF
# Backup Metadata

**Backup File:** $BACKUP_FILE
**Created:** $(date +"%Y-%m-%d %H:%M:%S")
**Database:** ntsb_aviation
**Backup Type:** $BACKUP_TYPE
**Duration:** ${DURATION}s

---

## Database State

- **Size:** $DB_SIZE
- **Total Rows:** $TOTAL_ROWS
- **Tables:** $TABLE_COUNT

---

## Backup Details

- **Compressed Size:** $BACKUP_SIZE
- **Uncompressed Size:** $UNCOMPRESSED_SIZE
- **Compression Ratio:** ${COMPRESSION_RATIO}%
- **MD5 Checksum:** ${CHECKSUM:-N/A}
- **Throughput:** $(echo "scale=1; $BACKUP_SIZE_BYTES / $DURATION / 1024 / 1024" | bc 2>/dev/null || echo "N/A") MB/s

---

## Restore Commands

### Full Restore (DESTRUCTIVE - drops existing database)

\`\`\`bash
# Extract backup
gunzip -c $BACKUP_FILE > /tmp/restore.sql

# Drop and recreate database
sudo -u postgres psql -c "DROP DATABASE IF EXISTS ntsb_aviation;"
sudo -u postgres psql -c "CREATE DATABASE ntsb_aviation OWNER $USER;"

# Restore
psql -d ntsb_aviation -f /tmp/restore.sql

# Cleanup
rm /tmp/restore.sql
\`\`\`

### Safe Restore (to new database for testing)

\`\`\`bash
# Create test database
sudo -u postgres psql -c "CREATE DATABASE ntsb_aviation_test OWNER $USER;"

# Restore to test database
gunzip -c $BACKUP_FILE | psql -d ntsb_aviation_test

# Verify
psql -d ntsb_aviation_test -c "SELECT COUNT(*) FROM events;"
\`\`\`

---

## Backup Verification

\`\`\`bash
# Test gzip integrity
gzip -t $BACKUP_FILE

# View first 100 lines
zcat $BACKUP_FILE | head -100

# Count lines
zcat $BACKUP_FILE | wc -l

# Search for specific table
zcat $BACKUP_FILE | grep "CREATE TABLE events"
\`\`\`

---

**Generated by:** /backup-db command
**Command Version:** 1.0
EOF

echo "✅ Metadata file: $METADATA_FILE"
echo ""
```

---

### PHASE 9: COMPLETION SUMMARY

```bash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ DATABASE BACKUP COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💾 BACKUP SUMMARY"
echo "   Type: $BACKUP_TYPE"
echo "   File: $BACKUP_FILE"
echo "   Size: $BACKUP_SIZE (compressed)"
echo "   Duration: ${DURATION}s"
echo "   Status: ✅ Success"
echo ""
echo "📊 DATABASE STATE"
echo "   Database size: $DB_SIZE"
echo "   Total rows: $TOTAL_ROWS"
echo "   Tables: $TABLE_COUNT"
echo ""
echo "📝 ARTIFACTS"
echo "   Backup file: $BACKUP_FILE"
echo "   Metadata: $METADATA_FILE"
[ -n "$CHECKSUM" ] && echo "   MD5 checksum: $CHECKSUM"
echo ""
echo "📋 NEXT STEPS"
echo "   1. Verify backup: gzip -t $BACKUP_FILE"
echo "   2. Store securely: Copy to cloud storage or external drive"
echo "   3. Test restore: /restore-db $BACKUP_FILE (to test database)"
echo "   4. Document: Add to backup log"
echo ""
echo "🔍 RESTORE COMMAND"
echo "   gunzip -c $BACKUP_FILE | psql -d ntsb_aviation_test"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
```

---

## SUCCESS CRITERIA

- [ ] pg_dump available
- [ ] Database connection verified
- [ ] Backup created successfully
- [ ] Backup compressed with gzip
- [ ] Backup integrity verified (if --verify)
- [ ] Metadata file generated
- [ ] Retention policy applied (if --keep)
- [ ] Completion summary displayed

---

## OUTPUT/DELIVERABLES

**Backup Files:**
- `backups/ntsb_aviation_[TYPE]_[TIMESTAMP].sql.gz` - Compressed SQL dump
- `backups/ntsb_aviation_[TYPE]_[TIMESTAMP].metadata.txt` - Backup metadata

---

## RELATED COMMANDS

- `/restore-db` - Restore database from backup
- `/validate-schema` - Validate database after restore
- `/sprint-status` - Check database status

---

## NOTES

### When to Use

**Regular Schedule:**
- Daily automated backups (cron/systemd timer)
- Before major operations (schema changes, large data loads)
- Weekly full backups
- Monthly archival backups

**Ad-Hoc:**
- Before testing destructive operations
- Before upgrades or migrations
- For sharing database state
- Disaster recovery preparation

### Backup Strategy Recommendations

**3-2-1 Rule:**
- **3** copies of data (production + 2 backups)
- **2** different storage media (local + cloud)
- **1** off-site backup (cloud storage)

**Retention Policy:**
- Keep 7 daily backups
- Keep 4 weekly backups
- Keep 12 monthly backups
- Archive yearly backups indefinitely

### Storage Considerations

**Local Storage:**
- Use SSD for fast backups
- Ensure sufficient disk space (2-3x database size)
- Use separate partition for backups

**Cloud Storage:**
- AWS S3, Google Cloud Storage, Azure Blob
- Encrypt backups before upload
- Use versioning for safety
- Consider Glacier for long-term archival

### Performance

**Backup Times (966 MB database):**
- Full backup: 2-3 minutes
- Schema only: <30 seconds
- Data only: 2-3 minutes

**Compression:**
- Typical ratio: 10-20% of uncompressed size
- ~966 MB database → ~100-200 MB compressed

---

## TROUBLESHOOTING

### Problem: "pg_dump: error: connection to server failed"

**Solution:**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify connection
psql -d ntsb_aviation -c "SELECT 1;"

# Check pg_hba.conf permissions
```

### Problem: "Permission denied writing to backups/"

**Solution:**
```bash
# Ensure directory exists and is writable
mkdir -p backups/
chmod 755 backups/

# Check disk space
df -h .
```

### Problem: "Backup file seems corrupted"

**Solution:**
```bash
# Test gzip integrity
gzip -t backups/ntsb_aviation_*.sql.gz

# If corrupted, re-run backup
/backup-db --verify

# Check disk for errors
sudo smartctl -a /dev/sda  # Adjust device as needed
```

### Problem: "Backup too slow"

**Solution:**
```bash
# Use custom format (faster for large databases)
pg_dump -d ntsb_aviation -Fc > backup.dump

# Use parallel dump (if PostgreSQL 9.3+)
pg_dump -d ntsb_aviation -Fd -j 4 -f backup_dir/

# Backup to faster disk (SSD)
```

---

## EXAMPLE USAGE

### Daily Full Backup

```bash
# Full backup with verification
/backup-db full --verify --keep 7

# Automated (add to crontab)
0 2 * * * cd /home/parobek/Code/NTSB_Datasets && /backup-db full --keep 7
```

### Schema-Only Backup (Fast)

```bash
# Quick schema backup before changes
/backup-db schema --name pre_migration

# Restore schema only
gunzip -c backups/pre_migration_schema_*.sql.gz | psql -d ntsb_aviation_test
```

### Data-Only Backup

```bash
# Backup data only (useful for data migrations)
/backup-db data --name data_snapshot

# Restore data only (assumes schema exists)
gunzip -c backups/data_snapshot_data_*.sql.gz | psql -d ntsb_aviation
```

### Custom Named Backup

```bash
# Backup before major operation
/backup-db full --name before_pre1982_integration --verify

# Document purpose in metadata
```

### Complete Workflow

```bash
# 1. Create backup
/backup-db full --verify --keep 7

# 2. Upload to cloud (example: AWS S3)
aws s3 cp backups/ntsb_aviation_full_*.sql.gz s3://my-backups/ntsb/

# 3. Test restore to temporary database
sudo -u postgres psql -c "CREATE DATABASE ntsb_test OWNER $USER;"
gunzip -c backups/ntsb_aviation_full_*.sql.gz | psql -d ntsb_test

# 4. Verify restore
psql -d ntsb_test -c "SELECT COUNT(*) FROM events;"

# 5. Cleanup test database
sudo -u postgres psql -c "DROP DATABASE ntsb_test;"
```

---

**Command Version:** 1.0
**Last Updated:** 2025-11-06
**Priority:** HIGH - Essential for data protection
