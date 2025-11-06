# Create Sub-Agent Task

Generate a new sub-agent tool task (run separately), to: $*

---

## PROJECT CONTEXT (NTSB Aviation Accident Database)
**Type:** PostgreSQL database repository for aviation accident investigation data
**Version:** Phase 1 Sprint 2 (95% complete)
**Status:** Historical data integration and query optimization complete
**Database:** PostgreSQL 18.0, 92,771 events (1977-2025), 726,969 rows, 966 MB
**Owner:** parobek (NO SUDO operations required)

## DATABASE STATE (Current)
- **events:** 92,771 rows (1977-2025, 48 years with gaps)
- **aircraft:** 94,533 rows (multiple aircraft per event)
- **flight_crew:** 31,003 rows
- **injury:** 169,337 rows
- **findings:** 69,838 rows
- **Load tracking:** avall.mdb (completed), Pre2008.mdb (completed), PRE1982.MDB (pending)
- **Materialized views:** 6 active (yearly, state, aircraft, decade, crew, findings)
- **Indexes:** 59 total (30 base + 29 performance/MV)

## QUALITY STANDARDS (Apply to ALL work)
✅ **Python Code:** ruff format + ruff check passing, type hints, PEP 8
✅ **SQL Code:** Syntax validated, no dangerous operations (unqualified DROP/TRUNCATE)
✅ **Bash Scripts:** shellcheck passing, error handling (set -e), proper quoting
✅ **Documentation:** Technical accuracy, consistent formatting, no broken links
✅ **Database Safety:** NO SUDO principle, regular user ownership, backup before destructive ops
✅ **Commits:** Conventional commits with context, impact, and affected tables/files

## SYSTEMATIC APPROACH (Work Through Phases)
1. **ANALYZE** - Read schema, data, understand current state, identify scope
2. **PLAN** - Break into subtasks, identify dependencies, estimate data volume/time
3. **EXECUTE** - Implement systematically, test with small datasets first, document decisions
4. **VERIFY** - Run /validate-schema, /data-quality, /benchmark (as appropriate)
5. **DOCUMENT** - Update README/CHANGELOG/CLAUDE.local.md, create deliverables report
6. **REPORT** - Provide comprehensive summary with metrics, query results, next steps

## COMMUNICATION GUIDELINES
- **Progress Updates:** Report after each major phase (not just at end)
- **Decisions:** Explain rationale for schema/query/ETL design choices
- **Issues:** Report blockers immediately (connection failures, permission errors)
- **Questions:** Ask for clarification if data requirements unclear
- **Metrics:** Include concrete numbers (rows affected, query latency, database size)

## DOCUMENTATION REQUIREMENTS
- **Update CHANGELOG.md** if changes affect schema/data/features
- **Update README.md** if changes affect statistics/usage/setup
- **Update CLAUDE.local.md** with session summary, database metrics, decisions
- **Update .claude/README.md** if new commands created
- **Document migrations** in scripts/migrations/ with README
- **Log major decisions** in CLAUDE.local.md (schema changes, ETL logic, performance)

## VERIFICATION CHECKLIST (Before Completion)
✅ All subtasks completed successfully
✅ Database validation passed: `/validate-schema`
✅ Data quality checks passed (if data changed): `/data-quality`
✅ Performance acceptable (if queries changed): `/benchmark quick`
✅ Documentation updated (if schema/data changed)
✅ Memory banks updated with session info
✅ No TODO/FIXME/WIP markers left uncommitted
✅ Cross-references validated (links, table names, file references)
✅ Git history clean (proper git mv for renames, no credentials committed)
✅ No temporary files left in /tmp/NTSB_Datasets/

## DELIVERABLES (Always Provide)
1. **Summary Report:** What was accomplished, database metrics, key decisions
2. **File Changes Log:** Files created/modified/deleted with line counts
3. **Database Changes:** Tables/indexes/views affected, row counts, size changes
4. **Query Results:** Performance metrics, data quality results, validation output
5. **Issues Encountered:** Problems found and how resolved (connection, data, permissions)
6. **Verification Results:** /validate-schema, /data-quality, /benchmark outputs
7. **Next Steps:** Recommendations for follow-up work (missing data, optimizations)
8. **Git Status:** Ready to commit? Any conflicts? Staging status

## ERROR HANDLING
- **Database Connection Errors:** Report error, check PostgreSQL status, verify credentials
- **Data Quality Issues:** Report specific problems (NULLs, outliers, duplicates), suggest fixes
- **Query Performance Issues:** Report slow queries, provide EXPLAIN ANALYZE, suggest indexes
- **Permission Errors:** Report affected operations, verify ownership with `\dt`
- **Schema Conflicts:** Report incompatibilities, suggest migrations
- **ETL Failures:** Report stage (extract/transform/load), row count mismatches, provide logs
- **Unclear Requirements:** Ask for clarification before proceeding (data formats, thresholds)

## EFFICIENCY GUIDELINES
- **Read Before Write:** Always read schema/data files before editing
- **Test with Small Data:** Test queries/ETL with LIMIT 100 first, then scale up
- **Bulk Operations:** Use COPY instead of INSERT for large data loads
- **Parallel Reads:** Use multiple Read tool calls in parallel when reading multiple files
- **Avoid Redundancy:** Check if work already done (query pg_tables, read existing scripts)
- **Staging Tables:** Use staging schema for complex ETL (prevents production corruption)
- **Smart Skipping:** If only docs changed, skip database validation checks

## MEMORY BANK UPDATES (Critical)
**Add to CLAUDE.local.md after completion:**
- **Date:** Current date (YYYY-MM-DD)
- **Task:** Brief description of work completed
- **Duration:** Approximate time spent
- **Database State:** Events count, total rows, database size
- **Key Decisions:** Schema changes, ETL logic, query optimization rationale
- **Metrics:** Tables modified, indexes added, query performance improvements
- **Data Quality:** Issues found/resolved, validation results
- **Issues:** Problems encountered and resolutions
- **Status:** Current sprint status after changes
- **Next Actions:** Recommended follow-up tasks (load PRE1982, optimize further)

## PROJECT-SPECIFIC NOTES
- **Scripts:** Organize by purpose (schema.sql, load_*.py, validate_*.sql, optimize_*.sql)
- **Migrations:** scripts/migrations/ with date prefix (YYYY-MM-DD_description.sql)
- **Documentation:** Major docs in docs/ (PRE1982_ANALYSIS.md, TOOLS_AND_UTILITIES.md)
- **Reports:** Sprint reports in root (SPRINT_1_REPORT.md, SPRINT_2_COMPLETION_REPORT.md)
- **Temporary Files:** Always use /tmp/NTSB_Datasets/ (not /tmp/ directly)
- **Data Files:** Never commit .csv, .mdb, .accdb files (use .gitignore)
- **Credentials:** Never commit .env, credentials.json, connection strings

## DATABASE-SPECIFIC GUIDELINES

### Schema Changes
- Always backup first: `/backup-db schema`
- Test in transaction: `BEGIN; ... ROLLBACK;` before `COMMIT;`
- Document in migration script with comments
- Update scripts/schema.sql to reflect changes
- Run `/validate-schema` after applying

### ETL Operations
- Use staging tables (staging schema) for complex loads
- Check load_tracking table to prevent duplicate loads
- Log progress (print row counts every 1000 rows)
- Verify foreign key integrity after load
- Run `/data-quality` to validate loaded data

### Query Optimization
- Benchmark BEFORE optimization: `/benchmark`
- Use EXPLAIN ANALYZE to identify bottlenecks
- Create indexes on filtered/joined columns
- Consider materialized views for expensive aggregations
- Benchmark AFTER optimization, compare results

### Data Quality
- Validate coordinates (-90/90 latitude, -180/180 longitude)
- Validate dates (1962-present for this dataset)
- Validate ages (10-120 years for crew)
- Check for NULL values in critical fields
- Detect outliers (statistical methods or domain knowledge)

## MCP TOOLS REFERENCE

### File Operations
- `mcp__MCP_DOCKER__read_file` - Read file contents
- `mcp__MCP_DOCKER__write_file` - Create/overwrite files
- `mcp__MCP_DOCKER__list_directory` - List directory contents
- `mcp__MCP_DOCKER__create_directory` - Create directories
- `mcp__MCP_DOCKER__search_files` - Search for files by pattern

### Database Operations (via Bash tool)
- `psql -d ntsb_aviation -c "SELECT ..."`
- `psql -d ntsb_aviation -f script.sql`
- `psql -d ntsb_aviation -c "\dt"` - List tables
- `psql -d ntsb_aviation -c "\d+ events"` - Describe table

### Web Search
- `mcp__MCP_DOCKER__brave_web_search` - Research best practices, error messages

### GitHub (if applicable)
- `mcp__MCP_DOCKER__get_file_contents` - Read GitHub files
- `mcp__MCP_DOCKER__create_pull_request` - Create PR

## CUSTOM COMMAND INTEGRATION

After completing tasks, suggest relevant commands:
- `/validate-schema` - After schema changes
- `/data-quality` - After data loading/transformation
- `/benchmark` - After query optimization
- `/refresh-mvs` - After data changes affecting materialized views
- `/cleanup-staging` - After ETL completion
- `/backup-db` - Before risky operations
- `/export-sample` - For testing/demos
- `/stage-commit` - To commit changes

## COMMON DATABASE SUB-AGENT PATTERNS

### Pattern 1: Schema Migration
```
Task: Add index to improve query performance

Phases:
1. Analyze current queries (EXPLAIN ANALYZE)
2. Design index strategy (composite vs single-column)
3. Create migration script (scripts/migrations/YYYY-MM-DD_*.sql)
4. Test in transaction (BEGIN; CREATE INDEX; ROLLBACK;)
5. Benchmark before/after
6. Apply migration (COMMIT)
7. Update schema.sql
8. Document in CHANGELOG.md
```

### Pattern 2: ETL Pipeline
```
Task: Load new data source

Phases:
1. Analyze source (examine schema, data types, volumes)
2. Create staging tables (staging schema)
3. Extract data (mdb-export or similar)
4. Transform data (clean, validate, deduplicate)
5. Load to staging (bulk COPY)
6. Validate data quality
7. Merge to production (INSERT SELECT)
8. Verify integrity
9. Document in load_tracking
```

### Pattern 3: Data Analysis
```
Task: Generate coverage analysis report

Phases:
1. Write analytical queries
2. Extract data to CSV
3. Calculate statistics
4. Generate visualizations
5. Write narrative report
6. Format as HTML/Markdown
7. Store in reports/
```

### Pattern 4: Query Optimization
```
Task: Improve materialized view performance

Phases:
1. Benchmark current performance
2. Analyze query plan (EXPLAIN ANALYZE)
3. Identify bottlenecks (sequential scans, sorts)
4. Design optimization (indexes, query rewrite)
5. Implement changes
6. Benchmark improved performance
7. Document improvements (X% faster)
```

## SUCCESS CRITERIA (Must Meet ALL)
✅ Task completed as specified (100% of requirements)
✅ No data corruption (referential integrity maintained)
✅ No regressions introduced (existing queries still work)
✅ Performance acceptable (query latencies within targets)
✅ Documentation comprehensive and accurate
✅ Memory banks updated with session
✅ All deliverables provided (reports, scripts, documentation)
✅ Clean git state (no conflicts, proper staging)
✅ Database validated (/validate-schema passed)
✅ Ready for user review/commit

## FINAL REMINDERS
⚠️ **ALWAYS read schema before changing** - understand table relationships
⚠️ **ALWAYS backup before destructive ops** - use /backup-db
⚠️ **ALWAYS test with small data first** - LIMIT 100, then scale up
⚠️ **ALWAYS validate after changes** - run /validate-schema, /data-quality
⚠️ **ALWAYS update memory banks** - critical for continuity
⚠️ **NEVER commit without user approval** - stage only, let user commit
⚠️ **NEVER use sudo** - regular user owns everything
⚠️ **NEVER commit data files** - only scripts and documentation
⚠️ **NEVER commit credentials** - use .gitignore, .env (not in repo)
⚠️ **ALWAYS provide comprehensive reports** - metrics, decisions, next steps
⚠️ **ALWAYS ask if unclear** - better to clarify than assume

## DATABASE-SPECIFIC REMINDERS
⚠️ **Check load_tracking** before loading historical data (prevent duplicates)
⚠️ **Use staging tables** for complex ETL (don't load directly to production)
⚠️ **Refresh materialized views** after data changes (they don't auto-update)
⚠️ **ANALYZE tables** after major changes (update query planner statistics)
⚠️ **Test migrations in transaction** (BEGIN; ... ROLLBACK; then COMMIT)
⚠️ **Document schema changes** in migration files and CHANGELOG
⚠️ **Preserve foreign key integrity** (load parents before children)

## TROUBLESHOOTING COMMON ISSUES

**"Permission denied for table"**
→ Check ownership: `\dt` (should show parobek as owner)
→ Solution: Run scripts/transfer_ownership.sql as postgres user

**"Database not found"**
→ Check databases: `\l`
→ Solution: Run scripts/setup_database.sh

**"Query too slow"**
→ Check query plan: `EXPLAIN ANALYZE SELECT ...`
→ Solution: Add indexes, create materialized views, rewrite query

**"Duplicate key violation"**
→ Check for duplicates: `SELECT ev_id, COUNT(*) FROM events GROUP BY ev_id HAVING COUNT(*) > 1;`
→ Solution: Use staging tables with deduplication logic

**"Foreign key violation"**
→ Check orphaned records: See scripts/validate_data.sql for queries
→ Solution: Load parent tables first, or use staging with INNER JOIN

**"Cannot extend file"**
→ Check disk space: `df -h`
→ Solution: Clean up /tmp/, vacuum database, or add disk space

---

**Now execute the task above with full context, systematic approach, and comprehensive deliverables.**
