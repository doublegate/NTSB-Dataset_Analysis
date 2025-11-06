/sub-agent THINK DEEPLY and execute comprehensive pre-commit workflow for database project:

## Phase 1: ANALYZE CHANGES (Understand Context)
- Run `git status` to identify all modified/new/deleted files
- Run `git diff --stat` to see scope of changes
- Identify which areas changed:
  - **scripts/** - ETL/data loading scripts (.py, .sql, .sh)
  - **docs/** - Documentation (.md files)
  - **.sql files** - Schema, queries, validation scripts
  - **data/** - CSV/data files (should not be committed)
  - **.env, credentials** - Sensitive files (must not be committed)
  - **Memory banks** - CLAUDE.md, CLAUDE.local.md
- Determine commit scope and impact level
- Check for uncommitted work in progress (TODO, FIXME, WIP comments)

## Phase 2: QUALITY VALIDATION (By File Type)

### Python Files (.py) - If Changed
- **Linting:** Run `ruff check .` to check for code quality issues
- **Formatting:** Run `ruff format --check .` to verify formatting
- **Suggest Fixes:** If issues found, show command: `ruff check . --fix && ruff format .`
- **Type Hints:** Verify type hints present in new functions
- **Documentation:** Check docstrings for new functions/classes

### SQL Files (.sql) - If Changed
- **Syntax Check:** Use `psql -d ntsb_aviation --dry-run -f <file>` to validate syntax
- **Dangerous Operations:** Check for:
  - `DROP TABLE` without confirmation
  - `TRUNCATE` without WHERE clause
  - `DELETE FROM` without WHERE clause (scan entire file)
- **Schema Validation:** If `scripts/schema.sql` changed:
  - Warn user that schema changes are critical
  - Suggest running `/validate-schema quick` after commit
- **Documentation:** Ensure complex queries have comments

### Markdown Files (.md) - If Changed
- **Link Validation:** Check for broken internal links (grep for `](` patterns)
- **Code Blocks:** Verify all code blocks have language tags (```sql, ```python, ```bash)
- **TODO/FIXME:** Scan for TODO/FIXME markers (warn only, don't fail)
- **Consistency:** Check for consistent heading levels (# ## ### ####)
- **Tables:** Verify markdown tables are properly formatted

### Bash Scripts (.sh) - If Changed
- **Shellcheck:** Run `shellcheck <file>` if available (warn if not installed)
- **Shebang:** Verify `#!/bin/bash` or `#!/usr/bin/env bash` present
- **Error Handling:** Check for `set -e` or equivalent error handling
- **User Permissions:** Verify no hardcoded usernames (should use `$USER`)

### Database-Specific Files
- **schema.sql:** Critical - verify no accidental drops/truncates
- **load_with_staging.py:** Verify load_tracking logic intact
- **validate_data.sql:** Ensure validation queries are sound
- **optimize_queries.sql:** Check materialized view definitions

## Phase 3: ENVIRONMENT VERIFICATION

### Check Active Environment
- **Python Environment:** If .py files changed, verify venv activated:
  ```bash
  if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠ Python virtual environment not activated"
    echo "Run: source .venv/bin/activate"
  fi
  ```
- **PostgreSQL Connection:** If schema/data files changed, test connection:
  ```bash
  psql -d ntsb_aviation -c "SELECT 1" > /dev/null 2>&1 || echo "⚠ PostgreSQL not accessible"
  ```

### Check Database State (If Schema/Data Changes)
- **Active Connections:** Warn if database has active connections during schema changes
- **Pending Migrations:** Check if load_tracking table shows incomplete loads
- **Staging Tables:** Warn if staging tables not empty (suggest `/cleanup-staging`)

## Phase 4: GITIGNORE MAINTENANCE
- Review .gitignore for missing patterns
- Add entries for:
  - **Temporary files:** `*.tmp`, `*.bak`, `*.swp`, `*~`
  - **Data files:** `*.csv` (except samples), `*.mdb`, `*.accdb`
  - **Logs:** `*.log`, `/logs/`
  - **Exports:** `/exports/`, `/tmp/NTSB_Datasets/`
  - **IDE files:** `.vscode/`, `.idea/`, `*.pyc`, `__pycache__/`
  - **OS files:** `.DS_Store`, `Thumbs.db`
  - **Credentials:** `.env`, `credentials.json`, `*.pem`, `*.key`
  - **Backups:** `backups/*.sql.gz` (should be in separate backup directory)
- Ensure /tmp/ files not committed
- **CRITICAL:** Verify no sensitive files staged:
  - `.env` files with database credentials
  - `credentials.json` or similar
  - Large data files (>10MB) - warn user
  - Personal notes or temporary testing files

## Phase 5: DATABASE-SPECIFIC SAFETY CHECKS

### Schema Change Detection
- If `scripts/schema.sql` modified:
  - **Alert:** "⚠ CRITICAL: Schema changes detected!"
  - **Recommend:** Create backup before applying: `/backup-db schema`
  - **Document:** Ensure CHANGELOG.md documents schema changes
  - **Migration:** Suggest creating migration script if needed

### Load Tracking Integrity
- If `scripts/load_with_staging.py` modified:
  - **Verify:** load_tracking table logic intact
  - **Test:** Suggest running with `--dry-run` flag first
  - **Document:** Update CHANGELOG with ETL changes

### Data File Prevention
- **Block:** Any `.csv` files in data/ directory (except approved samples)
- **Block:** Any `.mdb` or `.accdb` database files
- **Warn:** Files larger than 10MB (suggest Git LFS if truly needed)

### Credentials Detection
- **Scan:** All staged files for potential credentials:
  - Database passwords
  - API keys
  - Connection strings with embedded credentials
- **Pattern Match:** Look for `PASSWORD=`, `API_KEY=`, `SECRET=`
- **Block:** If found, abort and show how to remove

## Phase 6: DOCUMENTATION UPDATES

### docs/ Analysis
- **Read:** All modified .md files in docs/
- **Verify:** Technical accuracy matches code changes
- **Check:** Cross-references to other docs are valid
- **Update:** Version numbers if code versioned
- **Consistency:** Ensure terminology consistent with other docs

### README.md (Root)
- **Database Metrics:** Update if data loaded:
  - Total events count
  - Date range coverage (e.g., "1977-2025")
  - Database size
  - Table row counts
- **Feature Updates:** If new scripts/features added:
  - Add to "Scripts" section
  - Update usage examples
  - Document new capabilities
- **Status:** Update "Project Status" if sprint/phase changed
- **Performance:** Update metrics if optimization done

### CHANGELOG.md
- **Add Entry:** Comprehensive entry to [Unreleased] section
- **Format:** Follow convention:
  ```markdown
  ### [Unreleased]

  #### Data (for data loading/ETL changes)
  - Added: description
  - Changed: description
  - Fixed: description

  #### Schema (for schema changes)
  - Added: new tables/columns/indexes
  - Changed: modified constraints/types
  - Fixed: corrected schema issues

  #### Query (for query optimization)
  - Improved: query performance details
  - Added: new materialized views

  #### ETL (for ETL pipeline changes)
  - Enhanced: loader features
  - Fixed: duplicate detection issues

  #### Docs (for documentation)
  - Added: new guides/examples
  - Updated: outdated information

  #### Features (for new features)
  - Added: feature descriptions

  #### Fixes (for bug fixes)
  - Fixed: bug descriptions with issue refs
  ```
- **Include:** What changed, why, impact, related issues
- **List:** Affected files/tables/queries
- **Breaking Changes:** Document prominently if any

### QUICKSTART_POSTGRESQL.md
- Update if setup process changed
- Verify installation commands current
- Update troubleshooting if new issues discovered

### Sprint Documentation
- **CLAUDE.local.md:** Update current sprint status if major milestone reached
- **Sprint Reports:** Note if sprint report needs updating

## Phase 7: MEMORY BANK OPTIMIZATION

### Read Current State
- **Read:** CLAUDE.md (project guidance)
- **Read:** CLAUDE.local.md (session history, current status)
- **Read:** ~/.claude/CLAUDE.md (user patterns) if relevant
- **Identify:** Stale/duplicate/verbose content

### Update with Session Info
- **Add:** Current session summary to CLAUDE.local.md
- **Update Metrics:**
  - Database size
  - Row counts by table
  - Load tracking status
  - Sprint completion percentage
  - Date coverage (year range)
- **Record Decisions:** Key architectural decisions made
- **Update Issues:** Known Issues section with new discoveries
- **Next Actions:** Add follow-up tasks if applicable

### Optimize (Inline /mem-reduce Logic)
- **Remove:** Completed tasks from CLAUDE.local.md
- **Compress:** Verbose prose (20-30% reduction target):
  - Convert paragraphs to bullet points
  - Use tables for structured data
  - Remove redundant explanations
- **Consolidate:** Duplicate information between files
- **Archive:** Old session details if >10 sessions ago (move to separate archive file)
- **Preserve:** Critical information:
  - "NO SUDO" principle
  - Database ownership model
  - Sprint deliverables
  - Troubleshooting patterns

### Database-Specific Memory Bank Updates
- **Update:** Database state metrics (row counts, size, coverage)
- **Update:** Load tracking status (which databases loaded)
- **Update:** Materialized view status
- **Update:** Performance benchmarks if tested
- **Update:** Data quality metrics if assessed

### Verify
- **No Loss:** Ensure critical information preserved
- **Valid:** Cross-references still valid
- **Intact:** Quick reference sections complete and current

## Phase 8: CROSS-REFERENCE VALIDATION
- Check README links point to existing files
- Verify CHANGELOG references match actual changes
- Ensure docs/ cross-references valid
- Validate script references in documentation (e.g., references to scripts/ files)
- Check relative paths correct (especially in docs/)
- Verify command references valid (e.g., `/load-data`, `/validate-schema`)

## Phase 9: FINAL VERIFICATION

### File Count Check
```bash
echo "Total files tracked: $(git ls-files | wc -l)"
echo "Modified files: $(git diff --name-only | wc -l)"
echo "Staged changes: $(git diff --cached --name-only | wc -l)"
echo "Untracked files: $(git ls-files --others --exclude-standard | wc -l)"
```

### No Unintended Files
- **Verify:** No .swp, .tmp, .bak, *~ files staged
- **Check:** No large binary files added unintentionally (>10MB)
- **Confirm:** No debug/profiling artifacts staged
- **Verify:** No database dumps (.sql files with data)
- **Check:** No sensitive files (.env, credentials)

### Quality Gate (If Code Changed)
- **Python:** If .py files changed, ensure ruff checks passed (or documented why skipped)
- **SQL:** If .sql files changed, ensure syntax valid
- **Bash:** If .sh files changed, ensure shellcheck passed (if available)

### Database-Specific Final Checks
- If schema.sql changed: Recommend `/validate-schema quick` after commit
- If ETL scripts changed: Suggest testing with sample data
- If queries changed: Recommend `/benchmark` to verify performance
- If data quality scripts changed: Suggest running `/data-quality`

## Phase 10: STAGE ALL CHANGES
```bash
git add -A
```

**Display Staged Summary:**
```bash
git diff --cached --stat
git diff --cached --name-status
```

## Phase 11: CREATE COMPREHENSIVE COMMIT

### Commit Message Format
```
<type>(<scope>): <summary>

<detailed description>

## Changes Made
- Category 1: specific changes with context
- Category 2: specific changes with context

## Impact
- Performance: metrics/improvements if applicable
- Data: new data loaded/coverage extended if applicable
- Quality: validation improvements if applicable
- Features: new functionality if applicable
- Fixes: bugs resolved with issue references

## Files Modified
- path/to/file1 - brief description of changes
- path/to/file2 - brief description of changes

## Database State (if data/schema changed)
- Database size: X MB (was Y MB)
- Total events: X (was Y)
- Date coverage: YYYY-YYYY
- Tables affected: list tables

## Testing/Validation
- Schema validation: passed/not run
- Data quality: passed/not run
- Performance benchmarks: passed/not run
- Python linting: passed/not run

## Documentation
- README updated: yes/no
- CHANGELOG updated: yes/no
- docs/ updated: yes/no
- Memory banks updated: yes/no

## Related Commands
- Suggested follow-up: /validate-schema, /benchmark, etc.

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Commit Types (Database Project)
1. **data:** Data loading, ETL changes, data updates
2. **schema:** Database schema changes (tables, indexes, constraints)
3. **query:** Query optimization, materialized views
4. **etl:** ETL pipeline changes, staging logic
5. **docs:** Documentation only changes
6. **feat:** New features (new scripts, commands, capabilities)
7. **fix:** Bug fixes
8. **perf:** Performance improvements
9. **refactor:** Code restructuring without behavior change
10. **test:** Testing additions/changes
11. **chore:** Maintenance (dependencies, build, config)

### Scope Examples
- `schema` - Schema-related changes
- `etl` - ETL pipeline
- `loader` - Data loading scripts
- `validation` - Validation scripts
- `docs` - Documentation
- `commands` - Custom commands
- `backup` - Backup scripts
- `queries` - Query optimization
- `README` - README updates
- `CHANGELOG` - Changelog updates

### Requirements
- **Summary:** <72 chars, imperative mood ("add", "fix", "update", not "added", "fixed")
- **Body:** Detailed technical description with context
- **Lists:** Concrete changes with file names, not vague statements
- **Metrics:** Include numbers where relevant (row counts, performance, file sizes)
- **Attribution:** Always include Co-Authored-By
- **Database State:** Include metrics if data/schema changed
- **Testing:** Document what validation was done
- **Follow-up:** Suggest related commands to run next

## Phase 12: FINAL REVIEW (Before Commit)

### Display Commit Preview
- Show full commit message for review
- Show `git diff --cached --stat` (file summary)
- Show `git diff --cached --name-status` (change types)
- Highlight critical changes:
  - Schema changes (schema.sql)
  - ETL changes (load_with_staging.py)
  - Documentation updates (README, CHANGELOG)

### Confirm All Phases Completed
- ✅ Quality checks passed (or documented why skipped)
- ✅ Documentation updated
- ✅ Memory banks optimized
- ✅ No sensitive files staged
- ✅ Cross-references validated
- ✅ Database-specific checks complete

### User Confirmation
- **Ask:** "Ready to commit? Review the commit message and staged files above. (yes/no)"
- **Wait:** For explicit user confirmation
- **If yes:** Execute commit with message
- **If no:** Keep files staged, user can modify/abort

### Commit Execution
```bash
# Write commit message to temporary file
cat > /tmp/commit-message.txt << 'EOF'
[commit message here]
EOF

# Create commit
git commit -F /tmp/commit-message.txt

# Show commit summary
git log -1 --stat

# Clean up
rm /tmp/commit-message.txt
```

### Post-Commit Recommendations
Based on what was committed, suggest:
- **Schema changes:** "Run `/validate-schema` to verify database integrity"
- **Query changes:** "Run `/benchmark` to verify performance"
- **Data changes:** "Run `/data-quality` to assess data quality"
- **ETL changes:** "Test with `/load-data --dry-run <source>`"
- **Backup needed:** "Consider `/backup-db` before applying schema changes"

## SUCCESS CRITERIA

✅ All quality checks passed (or documented why skipped)
✅ All documentation updated and accurate
✅ Memory banks optimized and current
✅ No broken references or links
✅ Comprehensive commit message created
✅ All files properly staged
✅ No sensitive/temporary/data files included
✅ Database-specific safety checks passed
✅ User approved final commit
✅ Post-commit recommendations provided

## ERROR HANDLING

### Quality Check Failures
- **Python linting fails:** Abort, show errors, suggest `ruff check . --fix`
- **SQL syntax fails:** Abort, show line numbers, suggest manual review
- **Dangerous SQL detected:** Abort, highlight operations, require manual review

### Environment Issues
- **PostgreSQL not accessible:** Warn but continue (may not need DB for doc changes)
- **Python venv not activated:** Warn but continue if only non-Python changes

### Sensitive File Detection
- **Credentials found:** **ABORT IMMEDIATELY**, show file, show how to remove:
  ```bash
  git reset HEAD <file>
  echo "<file>" >> .gitignore
  git add .gitignore
  ```
- **Large files found:** Warn, suggest Git LFS if needed, ask to continue

### Documentation Issues
- **Broken links found:** Warn with list, continue (fix in separate commit if needed)
- **Missing changelog entry:** Warn strongly, continue (user can amend)

### Database State Issues
- **Staging tables not empty:** Warn, suggest `/cleanup-staging`, ask to continue
- **Schema mismatch detected:** Warn, recommend validation after commit

## OPTIMIZATION RULES

### Skip Quality Checks If:
- **Only .md files changed:** Skip Python/SQL validation (Phase 2)
- **Only memory banks changed:** Skip code quality checks
- **Only .gitignore changed:** Skip most validation

### Parallel Execution:
- Run independent checks concurrently where possible
- Don't wait for slow checks if they're not relevant

### User Experience:
- Show progress indicators for long-running checks
- Summarize results concisely (don't dump raw tool output)
- Provide actionable recommendations, not just errors
- Use color coding if terminal supports it (✅ ⚠ ❌)

## NOTES

### When to Use
- **Always:** Before committing changes (make this your standard workflow)
- **Especially:** Before schema changes, ETL updates, or major refactoring
- **Required:** Before sprint completion commits

### Quality Philosophy
- **Database-First:** Checks prioritize database integrity and data quality
- **Safety-First:** Block dangerous operations, require explicit confirmation
- **Documentation-First:** Ensure docs always match code state

### Breaking Changes
- **Always document** in commit footer with `BREAKING CHANGE:` prefix
- **Highlight** in CHANGELOG.md under Breaking Changes section
- **Consider** major version bump (semantic versioning)

### Integration with Workflow
- Use before `/daily-log` to ensure clean commit history
- Combine with `/backup-db` before risky changes
- Follow with `/validate-schema` or `/benchmark` as appropriate

## TROUBLESHOOTING

**Problem:** Ruff not found
**Solution:** Install in Python environment: `pip install ruff`

**Problem:** PostgreSQL connection failed
**Solution:** Start PostgreSQL: `sudo systemctl start postgresql`

**Problem:** Quality checks failing
**Solution:**
1. Fix issues shown in error messages
2. Or document why skipped and continue with `--force` (not implemented yet)

**Problem:** Sensitive file detected in staging
**Solution:**
```bash
# Remove from staging
git reset HEAD <file>

# Add to .gitignore
echo "<file>" >> .gitignore

# Stage .gitignore
git add .gitignore

# Re-run /stage-commit
```

**Problem:** Large file warning
**Solution:**
- If intentional and needed: Use Git LFS
- If unintentional: Remove from staging, add to .gitignore

**Problem:** Memory bank optimization failed
**Solution:** Skip optimization, continue with commit (memory banks can be updated later)

## EXAMPLES

### Example 1: Documentation-Only Change
```
/stage-commit

# Phases executed:
# - Phase 1: Analyze (detects only .md changed)
# - Phase 2: Quality (markdown validation only)
# - Phase 3: Environment (skipped - no code changes)
# - Phase 4: Gitignore (quick check)
# - Phase 5: Database checks (skipped)
# - Phase 6: Documentation (verify links, formatting)
# - Phase 7: Memory bank (update CLAUDE.local.md)
# - Phase 8: Cross-references (validate links)
# - Phase 9: Final verification
# - Phase 10: Stage files
# - Phase 11: Create commit
# - Phase 12: User confirmation

Commit created:
docs(README): update database statistics with Sprint 2 metrics

Updated row counts, date coverage, and database size
after Pre2008.mdb integration.

## Changes Made
- Database metrics: Updated to reflect 92,771 total events
- Date coverage: Extended to 1977-2025 (was 2008-2025)
- Performance: Documented query optimization results

## Files Modified
- README.md - Updated statistics section

## Documentation
- README updated: yes
- CHANGELOG updated: no (minor update)
- docs/ updated: no
- Memory banks updated: yes

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Example 2: Schema Change
```
/stage-commit

# Phases executed: ALL

⚠ CRITICAL: Schema changes detected!
⚠ Recommend backup: /backup-db schema
⚠ Validation: Run /validate-schema after commit

Commit created:
schema(events): add composite index on (ev_year, ev_state) for performance

Added composite index idx_events_year_state to optimize
year-by-state queries which are common in analytical
workloads.

## Changes Made
- Index: Created idx_events_year_state (ev_year, ev_state)
- Performance: 10x speedup on state-year aggregations

## Impact
- Performance: Queries using year+state filters now <10ms (was ~100ms)
- Database: Index adds ~15MB to database size

## Files Modified
- scripts/schema.sql - Added composite index
- scripts/optimize_queries.sql - Updated index documentation
- CHANGELOG.md - Documented schema change

## Database State
- Database size: 981 MB (was 966 MB)
- Total events: 92,771 (unchanged)
- Tables affected: events (new index only)

## Testing/Validation
- Schema validation: not run yet (run /validate-schema)
- Performance benchmarks: not run yet (run /benchmark)

## Documentation
- README updated: no (index only)
- CHANGELOG updated: yes
- Memory banks updated: yes

## Related Commands
- Run /validate-schema to verify integrity
- Run /benchmark to measure performance impact
- Consider /backup-db before applying to production

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Example 3: ETL Pipeline Enhancement
```
/stage-commit

# Phases executed: ALL
# Python linting: PASSED (ruff check .)
# Python formatting: PASSED (ruff format --check .)

Commit created:
etl(loader): improve duplicate detection performance by 50%

Replaced nested loop duplicate detection with set-based
operations using pandas .isin() method. Reduces Pre2008.mdb
load time from 180s to 90s.

## Changes Made
- Algorithm: Changed from O(n²) nested loops to O(n) set operations
- Performance: 50% reduction in duplicate detection time
- Memory: Slightly higher memory usage (~10MB more) for set storage

## Impact
- Performance: Pre2008.mdb load: 90s (was 180s)
- Performance: avall.mdb load: 28s (was 45s)
- Features: No behavioral changes, same duplicate detection logic

## Files Modified
- scripts/load_with_staging.py - Optimized duplicate detection
- CLAUDE.local.md - Updated performance metrics

## Database State
- No database changes (ETL logic only)

## Testing/Validation
- Python linting: passed (ruff)
- Load tested: yes (both avall.mdb and Pre2008.mdb)
- Results verified: duplicate counts match previous implementation

## Documentation
- README updated: no (internal optimization)
- CHANGELOG updated: yes
- Memory banks updated: yes

## Related Commands
- Test with: /load-data --dry-run avall.mdb

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Example 4: Data Loading (After /load-data)
```
/stage-commit

# After running /load-data avall.mdb monthly update

Commit created:
data(monthly): load November 2025 NTSB data update

Loaded monthly avall.mdb update from NTSB website.
Added 127 new accident events from October 2025.

## Changes Made
- Data: Loaded 127 new events (October 2025)
- Coverage: Extended to 2025-10-31 (was 2025-09-30)

## Impact
- Data: 127 new accident events
- Database: +2.3 MB size increase

## Files Modified
- (no code changes, data load only)

## Database State
- Database size: 968 MB (was 966 MB)
- Total events: 92,898 (was 92,771)
- Date coverage: 1977-2025 (extended to Oct 2025)
- Tables affected: events (+127), aircraft (+132), injury (+342),
  findings (+89), narratives (+45), flight_crew (+53)

## Testing/Validation
- Schema validation: passed (/validate-schema quick)
- Data quality: passed (/data-quality quick)
- Load tracking: updated (avall.mdb loaded 2025-11-06)

## Documentation
- Memory banks updated: yes (database metrics)

## Related Commands
- Run /data-coverage to see updated coverage
- Run /benchmark to verify performance maintained

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

**IMPORTANT:**

1. **Be thorough but efficient** - Skip irrelevant phases for doc-only changes
2. **Always create detailed commit messages** - Include context, metrics, impact
3. **Always ask for user confirmation** - Never commit without explicit approval
4. **Prioritize database safety** - Block dangerous operations, require backups
5. **Maintain documentation** - Keep README, CHANGELOG, memory banks current
6. **Follow "NO SUDO" principle** - All operations as regular user (after setup)
7. **Provide actionable recommendations** - Tell user what to do next

This command is the **gateway to all commits** - make it comprehensive, safe, and user-friendly.
