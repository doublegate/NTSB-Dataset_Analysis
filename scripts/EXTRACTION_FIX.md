# Table Extraction Fix

## ✅ Problem Fixed

The `extract_all_tables.fish` script had a critical bug where it treated all table names as a single table name, creating terrible filenames like:
```
'Country ct_iaids ct_seqevt dt_events dt_Flight_Crew eADMSPUB_DataDictionary engines events Events_Sequence Flight_Crew flight_time injury NTSB_Admin Occurrences seq_of_events states aircraft dt_aircraft Findings narratives .csv'
```

## 🔍 Root Cause

Fish shell's `for` loop with `mdb-tables` output was the problem:
- `mdb-tables` returns all table names as a **space-separated list on one line**
- The original loop: `for table in (mdb-tables "$database")` treated this entire line as one item
- Result: Script tried to export a "table" with that entire space-separated string as its name

## ✅ Solution Applied

**Fixed both scripts:**
1. **extract_all_tables.fish** - Now properly splits table names and adds database prefix
2. **extract_table.fish** - Now properly validates and uses database prefix

**Key Changes:**
```fish
# Split the space-separated output into individual table names
set tables (mdb-tables "$database" | string split ' ')

# Add database prefix to filenames
set db_basename (basename "$database" .mdb)
set db_basename (basename "$db_basename" .MDB)
mdb-export "$database" "$table" > "data/$db_basename-$table.csv"
```

## 📁 New Naming Convention

Files are now named: `{database}-{table}.csv`

**Examples:**
- `avall-events.csv` - events table from avall.mdb
- `avall-aircraft.csv` - aircraft table from avall.mdb
- `Pre2008-events.csv` - events table from Pre2008.mdb
- `Pre2008-aircraft.csv` - aircraft table from Pre2008.mdb
- `PRE1982-tblFirstHalf.csv` - tblFirstHalf table from PRE1982.MDB

**Benefits:**
- ✅ Proper filenames without spaces or concatenation
- ✅ Easy to identify which database each table came from
- ✅ No conflicts when extracting from multiple databases
- ✅ Consistent naming across all extraction scripts

## 🚀 Usage

Now you can safely run:

```fish
# Extract from all three databases
./scripts/extract_all_tables.fish datasets/avall.mdb
./scripts/extract_all_tables.fish datasets/Pre2008.mdb
./scripts/extract_all_tables.fish datasets/PRE1982.MDB
```

Expected output in `data/` directory:
```
avall-events.csv
avall-aircraft.csv
avall-engines.csv
Pre2008-events.csv
Pre2008-aircraft.csv
Pre2008-engines.csv
PRE1982-tblFirstHalf.csv
PRE1982-tblSecondHalf.csv
...
```

## 📊 Verification

After extraction, verify with:

```fish
# Count extracted tables
ls data/*.csv | wc -l

# Check file sizes
ls -lh data/ | head -20

# View specific table
head data/avall-events.csv
```

## 🔗 Updated Documentation

- **scripts/README.md** - Updated with naming convention explanation
- Both extraction scripts now include proper Fish string splitting
