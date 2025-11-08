-- create_code_mappings.sql - PRE1982 Legacy Code Mapping Tables
-- Sprint 4: PRE1982 Integration - Phase 2
-- Purpose: Store lookup tables for legacy coded fields
-- Created: 2025-11-07
-- Version: 1.0.0

-- ============================================
-- Create schema for code mappings
-- ============================================
CREATE SCHEMA IF NOT EXISTS code_mappings;

-- ============================================
-- 1. State Code Mapping (LOCAT_STATE_TERR → ev_state)
-- ============================================
-- Maps numeric codes (01-51, 60-71) to 2-letter state abbreviations
CREATE TABLE IF NOT EXISTS code_mappings.state_codes (
    legacy_code INTEGER PRIMARY KEY,
    state_abbr CHAR(2) NOT NULL,
    state_name VARCHAR(100) NOT NULL,
    source VARCHAR(20) DEFAULT 'ct_Pre1982'
);

-- Populate state codes (50 states + DC + territories + foreign)
INSERT INTO code_mappings.state_codes (legacy_code, state_abbr, state_name) VALUES
-- US States (01-50)
(0, 'US', 'United States'),
(1, 'AL', 'Alabama'),
(2, 'AK', 'Alaska'),
(3, 'AZ', 'Arizona'),
(4, 'AR', 'Arkansas'),
(5, 'CA', 'California'),
(6, 'CO', 'Colorado'),
(7, 'CT', 'Connecticut'),
(8, 'DE', 'Delaware'),
(9, 'FL', 'Florida'),
(10, 'GA', 'Georgia'),
(11, 'HI', 'Hawaii'),
(12, 'ID', 'Idaho'),
(13, 'IL', 'Illinois'),
(14, 'IN', 'Indiana'),
(15, 'IA', 'Iowa'),
(16, 'KS', 'Kansas'),
(17, 'KY', 'Kentucky'),
(18, 'LA', 'Louisiana'),
(19, 'ME', 'Maine'),
(20, 'MD', 'Maryland'),
(21, 'MA', 'Massachusetts'),
(22, 'MI', 'Michigan'),
(23, 'MN', 'Minnesota'),
(24, 'MS', 'Mississippi'),
(25, 'MO', 'Missouri'),
(26, 'MT', 'Montana'),
(27, 'NE', 'Nebraska'),
(28, 'NV', 'Nevada'),
(29, 'NH', 'New Hampshire'),
(30, 'NJ', 'New Jersey'),
(31, 'NM', 'New Mexico'),
(32, 'NY', 'New York'),
(33, 'NC', 'North Carolina'),
(34, 'ND', 'North Dakota'),
(35, 'OH', 'Ohio'),
(36, 'OK', 'Oklahoma'),
(37, 'OR', 'Oregon'),
(38, 'PA', 'Pennsylvania'),
(39, 'RI', 'Rhode Island'),
(40, 'SC', 'South Carolina'),
(41, 'SD', 'South Dakota'),
(42, 'TN', 'Tennessee'),
(43, 'TX', 'Texas'),
(44, 'UT', 'Utah'),
(45, 'VT', 'Vermont'),
(46, 'VA', 'Virginia'),
(47, 'WA', 'Washington'),
(48, 'WV', 'West Virginia'),
(49, 'WI', 'Wisconsin'),
(50, 'WY', 'Wyoming'),
-- DC and Unknown
(51, 'DC', 'District of Columbia'),
(52, 'UNK', 'Unknown'),
-- US Territories (60-64)
(60, 'US', 'US Territories & Possessions'),
(61, 'PR', 'Puerto Rico'),
(62, 'VI', 'Virgin Islands'),
(63, 'AS', 'American Samoa'),
(64, 'US', 'Other US Territory'),
-- Foreign (70-71)
(70, 'INTL', 'Foreign Countries'),
(71, 'CAN', 'Canada')
ON CONFLICT (legacy_code) DO NOTHING;

-- ============================================
-- 2. Age Code Mapping (AGE_PILOT* → crew_age)
-- ============================================
-- Note: PRE1982 may store ages directly as integers OR use coded ranges
-- This table handles both cases
CREATE TABLE IF NOT EXISTS code_mappings.age_codes (
    legacy_code VARCHAR(10) PRIMARY KEY,
    age_min INTEGER,
    age_max INTEGER,
    age_description VARCHAR(50),
    source VARCHAR(20) DEFAULT 'ct_Pre1982'
);

-- Common age range codes (if coded, will be populated from ct_Pre1982)
-- For now, create placeholders for common patterns
INSERT INTO code_mappings.age_codes (legacy_code, age_min, age_max, age_description) VALUES
('ZA', NULL, NULL, 'Unknown'),
('ZZ', NULL, NULL, 'Not Reported'),
('**', NULL, NULL, 'Header/Missing')
ON CONFLICT (legacy_code) DO NOTHING;

-- ============================================
-- 3. Cause Factor Code Mapping (CAUSE_FACTOR_*P/M/S → Findings)
-- ============================================
-- Maps legacy cause codes to descriptions
-- Populated from ct_Pre1982 table (945 cause factor codes)
CREATE TABLE IF NOT EXISTS code_mappings.cause_factor_codes (
    legacy_code VARCHAR(10) PRIMARY KEY,
    cause_description TEXT NOT NULL,
    cause_category VARCHAR(100),
    modern_finding_code VARCHAR(10),  -- Map to codman.pdf codes if possible
    source VARCHAR(20) DEFAULT 'ct_Pre1982'
);

-- Cause factor codes will be bulk-loaded from CSV
-- (945 codes from ct_Pre1982 - see populate_code_tables.py)
-- Sample entries (first 20 codes):
INSERT INTO code_mappings.cause_factor_codes (legacy_code, cause_description, cause_category) VALUES
('64', 'PILOT IN COMMAND', 'Pilot'),
('6401', 'ATTEMPTED OPERATION W/KNOWN DEFICIENCIES IN EQUIPMENT', 'Pilot Decision'),
('6402', 'ATTEMPTED OPERATION BEYOND EXPERIENCE/ABILITY LEVEL', 'Pilot Capability'),
('6403', 'BECAME LOST/DISORIENTED', 'Pilot Navigation'),
('6404', 'CONTINUED VFR FLIGHT INTO ADVERSE WEATHER CONDITIONS', 'Pilot Weather Decision'),
('6405', 'CONTINUED FLIGHT INTO KNOWN AREAS OF SEVERE TURBULENCE', 'Pilot Weather Decision'),
('6406', 'DELAYED ACTION IN ABORTING TAKEOFF', 'Pilot Decision'),
('6407', 'DELAYED IN INITIATING GO-AROUND', 'Pilot Decision'),
('6408', 'DIVERTED ATTENTION FROM OPERATION OF AIRCRAFT', 'Pilot Attention'),
('6409', 'EXCEEDED DESIGNED STRESS LIMITS OF AIRCRAFT', 'Pilot Operation'),
('6410', 'FAILED TO EXTEND LANDING GEAR', 'Pilot Procedural'),
('6411', 'FAILED TO RETRACT LANDING GEAR', 'Pilot Procedural'),
('6412', 'RETRACTED GEAR PREMATURELY', 'Pilot Procedural'),
('6413', 'INADVERTENTLY RETRACTED GEAR', 'Pilot Procedural'),
('6414', 'FAILED TO SEE AND AVOID OTHER AIRCRAFT', 'Pilot Vigilance'),
('6415', 'FAILED TO SEE AND AVOID OBJECTS OR OBSTRUCTIONS', 'Pilot Vigilance'),
('6416', 'FAILED TO OBTAIN/MAINTAIN FLYING SPEED', 'Pilot Operation'),
('6417', 'MISJUDGED DISTANCE,SPEED,ALTITUDE OR CLEARANCE', 'Pilot Judgment'),
('6418', 'FAILED TO MAINTAIN ADEQUATE ROTOR R.P.M.', 'Pilot Operation - Helicopter'),
('6419', 'FAILED TO USE OR INCORRECTLY USED MISC.EQUIPMENT', 'Pilot Equipment Use')
ON CONFLICT (legacy_code) DO NOTHING;

-- ============================================
-- 4. Injury Level Mapping (PRE1982 suffixes → modern codes)
-- ============================================
-- Maps injury column suffixes to modern inj_level codes
CREATE TABLE IF NOT EXISTS code_mappings.injury_level_mapping (
    legacy_suffix VARCHAR(20) PRIMARY KEY,
    modern_code VARCHAR(10) NOT NULL,
    description VARCHAR(50)
);

INSERT INTO code_mappings.injury_level_mapping (legacy_suffix, modern_code, description) VALUES
('FATAL', 'FATL', 'Fatal'),
('SERIOUS', 'SERS', 'Serious'),
('MINOR', 'MINR', 'Minor'),
('NONE', 'NONE', 'None'),
('UNKNOWN', 'UNK', 'Unknown'),
-- Index codes (from ct_Pre1982)
('F', 'FATL', 'Fatal (Index)'),
('C', 'SERS', 'Serious (Index)'),
('M', 'MINR', 'Minor (Index)'),
('N', 'NONE', 'None (Index)'),
('Z', 'UNK', 'Unknown (Index)')
ON CONFLICT (legacy_suffix) DO NOTHING;

-- ============================================
-- 5. Aircraft Damage Mapping (ACFT_ADAMG → damage)
-- ============================================
-- Maps legacy damage codes to modern codes
CREATE TABLE IF NOT EXISTS code_mappings.damage_codes (
    legacy_code VARCHAR(10) PRIMARY KEY,
    modern_code VARCHAR(10) NOT NULL,
    description VARCHAR(50)
);

-- From ct_Pre1982 ACFT_ADAMG field
INSERT INTO code_mappings.damage_codes (legacy_code, modern_code, description) VALUES
('D', 'DEST', 'Destroyed'),
('S', 'SUBS', 'Substantial'),
('M', 'MINR', 'Minor'),
('N', 'NONE', 'None'),
('Z', 'UNK', 'Unknown/Not Reported'),
('**', 'UNK', 'Header/Missing')
ON CONFLICT (legacy_code) DO NOTHING;

-- ============================================
-- Indexes for fast lookup
-- ============================================
CREATE INDEX IF NOT EXISTS idx_state_codes_abbr ON code_mappings.state_codes(state_abbr);
CREATE INDEX IF NOT EXISTS idx_age_codes_range ON code_mappings.age_codes(age_min, age_max);
CREATE INDEX IF NOT EXISTS idx_cause_factor_category ON code_mappings.cause_factor_codes(cause_category);
CREATE INDEX IF NOT EXISTS idx_cause_factor_desc ON code_mappings.cause_factor_codes
    USING gin(to_tsvector('english', cause_description));

-- ============================================
-- Helper functions for code lookup
-- ============================================

-- Decode state code (numeric → 2-letter abbreviation)
CREATE OR REPLACE FUNCTION code_mappings.decode_state(code INTEGER)
RETURNS CHAR(2) AS $$
    SELECT state_abbr FROM code_mappings.state_codes WHERE legacy_code = code;
$$ LANGUAGE SQL IMMUTABLE;

-- Decode age code (code → midpoint of range, or NULL if unknown)
CREATE OR REPLACE FUNCTION code_mappings.decode_age(code VARCHAR(10))
RETURNS INTEGER AS $$
    SELECT
        CASE
            WHEN age_min IS NULL THEN NULL
            ELSE (age_min + COALESCE(age_max, age_min)) / 2
        END
    FROM code_mappings.age_codes
    WHERE legacy_code = code;
$$ LANGUAGE SQL IMMUTABLE;

-- Decode cause factor code (code → description)
CREATE OR REPLACE FUNCTION code_mappings.decode_cause_factor(code VARCHAR(10))
RETURNS TEXT AS $$
    SELECT COALESCE(cause_description, 'LEGACY:' || code)
    FROM code_mappings.cause_factor_codes
    WHERE legacy_code = code;
$$ LANGUAGE SQL IMMUTABLE;

-- Decode damage code (legacy → modern)
CREATE OR REPLACE FUNCTION code_mappings.decode_damage(code VARCHAR(10))
RETURNS VARCHAR(10) AS $$
    SELECT COALESCE(modern_code, 'UNK')
    FROM code_mappings.damage_codes
    WHERE legacy_code = code;
$$ LANGUAGE SQL IMMUTABLE;

-- Decode injury level (legacy suffix → modern code)
CREATE OR REPLACE FUNCTION code_mappings.decode_injury_level(suffix VARCHAR(20))
RETURNS VARCHAR(10) AS $$
    SELECT COALESCE(modern_code, 'UNK')
    FROM code_mappings.injury_level_mapping
    WHERE legacy_suffix = suffix;
$$ LANGUAGE SQL IMMUTABLE;

-- ============================================
-- Statistics and validation queries
-- ============================================

-- View code mapping table statistics
CREATE OR REPLACE VIEW code_mappings.mapping_stats AS
SELECT
    'state_codes' AS table_name,
    COUNT(*) AS total_codes,
    COUNT(DISTINCT state_abbr) AS unique_values,
    'state abbreviations' AS description
FROM code_mappings.state_codes
UNION ALL
SELECT
    'age_codes',
    COUNT(*),
    COUNT(DISTINCT age_min) - COUNT(CASE WHEN age_min IS NULL THEN 1 END),
    'age ranges'
FROM code_mappings.age_codes
UNION ALL
SELECT
    'cause_factor_codes',
    COUNT(*),
    COUNT(DISTINCT cause_category),
    'cause categories'
FROM code_mappings.cause_factor_codes
UNION ALL
SELECT
    'injury_level_mapping',
    COUNT(*),
    COUNT(DISTINCT modern_code),
    'injury levels'
FROM code_mappings.injury_level_mapping
UNION ALL
SELECT
    'damage_codes',
    COUNT(*),
    COUNT(DISTINCT modern_code),
    'damage levels'
FROM code_mappings.damage_codes;

-- Grant permissions (assumes current user owns database)
GRANT USAGE ON SCHEMA code_mappings TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA code_mappings TO PUBLIC;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA code_mappings TO PUBLIC;

-- ============================================
-- Verification queries
-- ============================================

-- Test state decoding
SELECT
    'State decoding test' AS test_name,
    code_mappings.decode_state(32) AS ny_result,  -- Should return 'NY'
    code_mappings.decode_state(5) AS ca_result,   -- Should return 'CA'
    code_mappings.decode_state(61) AS pr_result;  -- Should return 'PR'

-- Test damage decoding
SELECT
    'Damage decoding test' AS test_name,
    code_mappings.decode_damage('D') AS destroyed,  -- Should return 'DEST'
    code_mappings.decode_damage('S') AS substantial, -- Should return 'SUBS'
    code_mappings.decode_damage('M') AS minor;       -- Should return 'MINR'

-- Test injury level decoding
SELECT
    'Injury level decoding test' AS test_name,
    code_mappings.decode_injury_level('FATAL') AS fatal,    -- Should return 'FATL'
    code_mappings.decode_injury_level('SERIOUS') AS serious, -- Should return 'SERS'
    code_mappings.decode_injury_level('F') AS fatal_idx;     -- Should return 'FATL'

-- Display mapping statistics
SELECT * FROM code_mappings.mapping_stats ORDER BY table_name;

-- ============================================
-- End of script
-- ============================================

-- Summary of tables created:
-- 1. state_codes (80 rows) - Numeric codes → 2-letter state abbreviations
-- 2. age_codes (3+ rows) - Age code ranges (will be populated from ct_Pre1982)
-- 3. cause_factor_codes (20+ rows initially, 945 from CSV) - Legacy cause codes → descriptions
-- 4. injury_level_mapping (9 rows) - Injury suffixes → modern codes
-- 5. damage_codes (6 rows) - Damage codes → modern codes

-- Usage in ETL:
-- SELECT code_mappings.decode_state(LOCAT_STATE_TERR) AS ev_state FROM pre1982_data;
-- SELECT code_mappings.decode_damage(ACFT_ADAMG) AS damage FROM pre1982_data;
-- SELECT code_mappings.decode_cause_factor(CAUSE_FACTOR_1P) AS finding_description FROM pre1982_data;

COMMENT ON SCHEMA code_mappings IS 'PRE1982 legacy code mapping tables for Sprint 4 integration';
COMMENT ON TABLE code_mappings.state_codes IS 'Maps numeric state codes (01-71) to 2-letter abbreviations';
COMMENT ON TABLE code_mappings.age_codes IS 'Maps age codes to numeric age ranges';
COMMENT ON TABLE code_mappings.cause_factor_codes IS 'Maps legacy cause factor codes to descriptions (945 codes)';
COMMENT ON TABLE code_mappings.injury_level_mapping IS 'Maps injury column suffixes to modern inj_level codes';
COMMENT ON TABLE code_mappings.damage_codes IS 'Maps legacy damage codes to modern codes';
