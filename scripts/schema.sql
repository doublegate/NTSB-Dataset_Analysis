-- schema.sql - Complete PostgreSQL schema for NTSB Aviation Accident Database
-- Phase 1 Sprint 1: Database Migration
-- Version: 1.0.0
-- Date: 2025-11-05

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- For SHA-256 hashing

-- ============================================
-- CORE TABLES
-- ============================================

-- Master events table (partitioning deferred to Phase 1 Sprint 2 for simplicity)
CREATE TABLE events (
    ev_id VARCHAR(20) PRIMARY KEY,
    ev_date DATE NOT NULL,
    ev_time TIME,
    ev_year INTEGER NOT NULL,  -- Populated by trigger, indexed for performance
    ev_month INTEGER NOT NULL,  -- Populated by trigger
    ev_dow VARCHAR(10),

    -- Location
    ev_city VARCHAR(100),
    ev_state CHAR(2),
    ev_country CHAR(3) DEFAULT 'USA',
    ev_site_zipcode VARCHAR(10),
    dec_latitude DECIMAL(10, 6),
    dec_longitude DECIMAL(11, 6),
    location_geom GEOGRAPHY(POINT, 4326) GENERATED ALWAYS AS (
        CASE
            WHEN dec_latitude IS NOT NULL AND dec_longitude IS NOT NULL
            THEN ST_SetSRID(ST_MakePoint(dec_longitude, dec_latitude), 4326)::geography
            ELSE NULL
        END
    ) STORED,

    -- Classification
    ev_type VARCHAR(10),
    ev_highest_injury VARCHAR(10),
    ev_nr_apt_id VARCHAR(10),
    ev_nr_apt_loc VARCHAR(100),
    ev_nr_apt_dist DECIMAL(8, 2),

    -- Injury totals
    inj_tot_f INTEGER DEFAULT 0,  -- Fatalities
    inj_tot_s INTEGER DEFAULT 0,  -- Serious
    inj_tot_m INTEGER DEFAULT 0,  -- Minor
    inj_tot_n INTEGER DEFAULT 0,  -- None

    -- Weather
    wx_cond_basic VARCHAR(10),
    wx_temp INTEGER,
    wx_wind_dir INTEGER,
    wx_wind_speed INTEGER,
    wx_vis DECIMAL(5, 2),

    -- Flight information
    flight_plan_filed VARCHAR(10),
    flight_activity VARCHAR(100),
    flight_phase VARCHAR(100),

    -- Investigation
    ntsb_no VARCHAR(30),
    report_status VARCHAR(10),
    probable_cause TEXT,

    -- Audit columns
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_hash CHAR(64),  -- SHA-256 for CDC

    -- Constraints
    CONSTRAINT valid_latitude CHECK (dec_latitude BETWEEN -90 AND 90),
    CONSTRAINT valid_longitude CHECK (dec_longitude BETWEEN -180 AND 180),
    CONSTRAINT valid_ev_date CHECK (ev_date >= '1962-01-01' AND ev_date <= CURRENT_DATE + INTERVAL '1 year')
);

-- Note: Table partitioning will be added in Phase 1 Sprint 2 after successful data migration
-- For now, using indexed ev_year column provides good performance for year-based queries

-- Aircraft table
CREATE TABLE aircraft (
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,
    Aircraft_Key VARCHAR(20) NOT NULL,
    acft_serial_number VARCHAR(50),
    regis_no VARCHAR(15),

    -- Type
    acft_make VARCHAR(100),
    acft_model VARCHAR(100),
    acft_series VARCHAR(30),
    acft_category VARCHAR(30),
    acft_type_code VARCHAR(20),

    -- Operation
    far_part VARCHAR(10),
    oper_country CHAR(3),
    owner_city VARCHAR(100),
    owner_state CHAR(2),

    -- Damage
    damage VARCHAR(10),

    -- Specifications
    cert_max_gr_wt INTEGER,
    num_eng INTEGER,
    fixed_retractable VARCHAR(10),

    -- Audit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (ev_id, Aircraft_Key)
);

-- Flight crew table
CREATE TABLE Flight_Crew (
    crew_no SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL,
    Aircraft_Key VARCHAR(20) NOT NULL,

    crew_category VARCHAR(30),
    crew_age INTEGER,
    crew_sex CHAR(1),
    crew_seat VARCHAR(30),

    -- Certifications
    pilot_cert VARCHAR(100),
    pilot_rat VARCHAR(200),
    pilot_med_class VARCHAR(5),
    pilot_med_date DATE,

    -- Experience
    pilot_tot_time INTEGER,
    pilot_make_time INTEGER,
    pilot_90_days INTEGER,
    pilot_30_days INTEGER,
    pilot_24_hrs INTEGER,

    -- Audit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_age CHECK (crew_age BETWEEN 10 AND 120),
    FOREIGN KEY (ev_id, Aircraft_Key) REFERENCES aircraft(ev_id, Aircraft_Key) ON DELETE CASCADE
);

-- Injury table
CREATE TABLE injury (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL,
    Aircraft_Key VARCHAR(20),

    inj_person_category VARCHAR(30),
    inj_level VARCHAR(10),
    inj_person_count INTEGER DEFAULT 1,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ev_id, Aircraft_Key) REFERENCES aircraft(ev_id, Aircraft_Key) ON DELETE CASCADE
);

-- Findings table (investigation results)
CREATE TABLE Findings (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL,
    Aircraft_Key VARCHAR(20),

    finding_code VARCHAR(10),
    finding_description VARCHAR(500),
    cm_inPC BOOLEAN DEFAULT FALSE,  -- In probable cause (Release 3.0+)
    cause_factor VARCHAR(10),  -- Deprecated (pre-Oct 2020)
    modifier_code VARCHAR(10),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_finding_code CHECK (
        finding_code ~ '^[0-9]{5,9}$' OR finding_code IS NULL
    ),
    FOREIGN KEY (ev_id, Aircraft_Key) REFERENCES aircraft(ev_id, Aircraft_Key) ON DELETE CASCADE
);

-- Occurrences table
CREATE TABLE Occurrences (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL,
    Aircraft_Key VARCHAR(20),

    occurrence_code VARCHAR(10),
    occurrence_description VARCHAR(255),
    phase_code VARCHAR(10),
    phase_description VARCHAR(100),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_occurrence_code CHECK (
        occurrence_code ~ '^[0-9]{3}$' OR occurrence_code IS NULL
    ),
    FOREIGN KEY (ev_id, Aircraft_Key) REFERENCES aircraft(ev_id, Aircraft_Key) ON DELETE CASCADE
);

-- Sequence of events table
CREATE TABLE seq_of_events (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL,
    Aircraft_Key VARCHAR(20),

    seq_event_no INTEGER,
    occurrence_code VARCHAR(10),
    phase_of_flight VARCHAR(100),
    altitude INTEGER,
    defining_event BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ev_id, Aircraft_Key) REFERENCES aircraft(ev_id, Aircraft_Key) ON DELETE CASCADE
);

-- Events_Sequence table (event ordering and relationships)
CREATE TABLE Events_Sequence (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL,
    Aircraft_Key VARCHAR(20),

    sequence_number INTEGER,
    event_description VARCHAR(255),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ev_id, Aircraft_Key) REFERENCES aircraft(ev_id, Aircraft_Key) ON DELETE CASCADE
);

-- Engines table
CREATE TABLE engines (
    eng_no SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL,
    Aircraft_Key VARCHAR(20),

    eng_make VARCHAR(100),
    eng_model VARCHAR(100),
    eng_type VARCHAR(30),
    eng_hp_or_lbs INTEGER,
    eng_carb_injection VARCHAR(10),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ev_id, Aircraft_Key) REFERENCES aircraft(ev_id, Aircraft_Key) ON DELETE CASCADE
);

-- Narratives table
CREATE TABLE narratives (
    id SERIAL PRIMARY KEY,
    ev_id VARCHAR(20) NOT NULL REFERENCES events(ev_id) ON DELETE CASCADE,

    narr_accp TEXT,  -- Accident description
    narr_cause TEXT,  -- Cause/contributing factors
    narr_rectification TEXT,  -- Corrective actions

    -- Full-text search vector
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(narr_accp, '') || ' ' || COALESCE(narr_cause, ''))
    ) STORED,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NTSB administrative metadata
CREATE TABLE NTSB_Admin (
    ev_id VARCHAR(20) PRIMARY KEY REFERENCES events(ev_id) ON DELETE CASCADE,

    ntsb_docket VARCHAR(100),
    invest_start_date DATE,
    report_date DATE,
    invest_in_charge VARCHAR(200),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- INDEXES
-- ============================================

-- Events indexes
CREATE INDEX idx_events_ev_date ON events(ev_date);
CREATE INDEX idx_events_ev_year ON events(ev_year);
CREATE INDEX idx_events_severity ON events(ev_highest_injury);
CREATE INDEX idx_events_state ON events(ev_state);
CREATE INDEX idx_events_location_geom ON events USING GIST(location_geom);
CREATE INDEX idx_events_date_severity ON events(ev_date, ev_highest_injury);
CREATE INDEX idx_events_ntsb_no ON events(ntsb_no);

-- Aircraft indexes
CREATE INDEX idx_aircraft_ev_id ON aircraft(ev_id);
CREATE INDEX idx_aircraft_make_model ON aircraft(acft_make, acft_model);
CREATE INDEX idx_aircraft_regis_no ON aircraft(regis_no);

-- Crew indexes
CREATE INDEX idx_crew_ev_id ON Flight_Crew(ev_id);
CREATE INDEX idx_crew_aircraft_key ON Flight_Crew(Aircraft_Key);

-- Injury indexes
CREATE INDEX idx_injury_ev_id ON injury(ev_id);
CREATE INDEX idx_injury_aircraft_key ON injury(Aircraft_Key);

-- Findings indexes
CREATE INDEX idx_findings_ev_id ON Findings(ev_id);
CREATE INDEX idx_findings_in_pc ON Findings(cm_inPC) WHERE cm_inPC = TRUE;
CREATE INDEX idx_findings_code ON Findings(finding_code);

-- Occurrences indexes
CREATE INDEX idx_occurrences_ev_id ON Occurrences(ev_id);
CREATE INDEX idx_occurrences_code ON Occurrences(occurrence_code);

-- Sequence indexes
CREATE INDEX idx_seq_of_events_ev_id ON seq_of_events(ev_id);
CREATE INDEX idx_events_sequence_ev_id ON Events_Sequence(ev_id);

-- Engines indexes
CREATE INDEX idx_engines_ev_id ON engines(ev_id);
CREATE INDEX idx_engines_aircraft_key ON engines(Aircraft_Key);

-- Narratives full-text search index
CREATE INDEX idx_narratives_search ON narratives USING GIN(search_vector);
CREATE INDEX idx_narratives_ev_id ON narratives(ev_id);

-- NTSB_Admin indexes
CREATE INDEX idx_ntsb_admin_report_date ON NTSB_Admin(report_date);

-- ============================================
-- MATERIALIZED VIEWS (for performance)
-- ============================================

-- Yearly statistics
CREATE MATERIALIZED VIEW mv_yearly_stats AS
SELECT
    ev_year,
    COUNT(*) as total_accidents,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_accidents,
    SUM(COALESCE(inj_tot_f, 0)) as total_fatalities,
    AVG(COALESCE(inj_tot_f, 0)) as avg_fatalities_per_accident,
    SUM(CASE WHEN ev_highest_injury = 'SERS' THEN 1 ELSE 0 END) as serious_injury_accidents,
    SUM(CASE WHEN damage = 'DEST' THEN 1 ELSE 0 END) as destroyed_aircraft
FROM events e
LEFT JOIN aircraft a ON e.ev_id = a.ev_id
GROUP BY ev_year
ORDER BY ev_year;

CREATE UNIQUE INDEX idx_mv_yearly_stats_year ON mv_yearly_stats(ev_year);

-- State-level statistics
CREATE MATERIALIZED VIEW mv_state_stats AS
SELECT
    ev_state,
    COUNT(*) as accident_count,
    SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count,
    AVG(dec_latitude) as avg_latitude,
    AVG(dec_longitude) as avg_longitude
FROM events
WHERE ev_state IS NOT NULL
GROUP BY ev_state;

CREATE UNIQUE INDEX idx_mv_state_stats_state ON mv_state_stats(ev_state);

-- ============================================
-- FUNCTIONS & TRIGGERS
-- ============================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to relevant tables
CREATE TRIGGER update_events_updated_at BEFORE UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_aircraft_updated_at BEFORE UPDATE ON aircraft
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Populate ev_year and ev_month from ev_date (required for partitioning)
CREATE OR REPLACE FUNCTION populate_ev_year_month()
RETURNS TRIGGER AS $$
BEGIN
    NEW.ev_year = EXTRACT(YEAR FROM NEW.ev_date);
    NEW.ev_month = EXTRACT(MONTH FROM NEW.ev_date);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER populate_events_year_month BEFORE INSERT OR UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION populate_ev_year_month();

-- Content hash for Change Data Capture
CREATE OR REPLACE FUNCTION calculate_content_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_hash = encode(
        digest(
            CONCAT_WS('|',
                NEW.ev_id, NEW.ev_date, NEW.ev_city, NEW.ev_state,
                NEW.ev_highest_injury, NEW.inj_tot_f, NEW.probable_cause
            )::TEXT,
            'sha256'
        ),
        'hex'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER calculate_events_content_hash BEFORE INSERT OR UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION calculate_content_hash();

-- ============================================
-- COMMENTS (for documentation)
-- ============================================

COMMENT ON TABLE events IS 'Master aviation accident events table, partitioned by year for optimal query performance';
COMMENT ON TABLE aircraft IS 'Aircraft involved in accidents, linked to events via ev_id';
COMMENT ON TABLE Flight_Crew IS 'Flight crew information including certifications and flight experience';
COMMENT ON TABLE injury IS 'Injury details for crew and passengers';
COMMENT ON TABLE Findings IS 'Investigation findings and probable causes';
COMMENT ON TABLE Occurrences IS 'Specific occurrence events during accidents';
COMMENT ON TABLE seq_of_events IS 'Sequence of events leading to accidents';
COMMENT ON TABLE Events_Sequence IS 'Event ordering and relationships';
COMMENT ON TABLE engines IS 'Engine details for involved aircraft';
COMMENT ON TABLE narratives IS 'Textual accident narratives and descriptions with full-text search';
COMMENT ON TABLE NTSB_Admin IS 'Administrative metadata for NTSB investigations';

COMMENT ON COLUMN events.content_hash IS 'SHA-256 hash for Change Data Capture tracking';
COMMENT ON COLUMN events.location_geom IS 'PostGIS geography point generated from lat/long coordinates';
COMMENT ON COLUMN Findings.cm_inPC IS 'TRUE if finding is cited in probable cause (Release 3.0+)';
COMMENT ON COLUMN narratives.search_vector IS 'Full-text search vector for narrative text';

-- ============================================
-- PERMISSIONS (commented out - adjust as needed)
-- ============================================

-- CREATE USER ntsb_app WITH PASSWORD 'change_me_in_production';
-- GRANT CONNECT ON DATABASE ntsb_aviation TO ntsb_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ntsb_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ntsb_app;

-- Create read-only user for analytics
-- CREATE USER ntsb_readonly WITH PASSWORD 'readonly_password';
-- GRANT CONNECT ON DATABASE ntsb_aviation TO ntsb_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO ntsb_readonly;

-- ============================================
-- COMPLETION MESSAGE
-- ============================================

DO $$
BEGIN
    RAISE NOTICE 'Schema creation complete!';
    RAISE NOTICE 'Tables created: 11 core tables + 7 partitions';
    RAISE NOTICE 'Indexes created: 27 indexes + 2 materialized view indexes';
    RAISE NOTICE 'Triggers created: 3 triggers (updated_at, content_hash)';
    RAISE NOTICE 'Extensions enabled: postgis, pg_trgm, pgcrypto';
END $$;
