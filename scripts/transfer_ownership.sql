-- transfer_ownership.sql - Transfer ownership of all database objects to parobek
-- Run as: sudo -u postgres psql -d ntsb_aviation -f scripts/transfer_ownership.sql

\echo '============================================================'
\echo 'Transferring Ownership to parobek'
\echo '============================================================'
\echo ''

-- Transfer ownership of all tables in public schema
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public'
    LOOP
        EXECUTE 'ALTER TABLE public.' || quote_ident(r.tablename) || ' OWNER TO parobek';
        RAISE NOTICE 'Transferred table: %', r.tablename;
    END LOOP;
END
$$;

-- Transfer ownership of all sequences
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT sequence_name FROM information_schema.sequences WHERE sequence_schema = 'public'
    LOOP
        EXECUTE 'ALTER SEQUENCE public.' || quote_ident(r.sequence_name) || ' OWNER TO parobek';
        RAISE NOTICE 'Transferred sequence: %', r.sequence_name;
    END LOOP;
END
$$;

-- Transfer ownership of all views
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT viewname FROM pg_views WHERE schemaname = 'public'
    LOOP
        EXECUTE 'ALTER VIEW public.' || quote_ident(r.viewname) || ' OWNER TO parobek';
        RAISE NOTICE 'Transferred view: %', r.viewname;
    END LOOP;
END
$$;

-- Transfer ownership of all materialized views
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT matviewname FROM pg_matviews WHERE schemaname = 'public'
    LOOP
        EXECUTE 'ALTER MATERIALIZED VIEW public.' || quote_ident(r.matviewname) || ' OWNER TO parobek';
        RAISE NOTICE 'Transferred materialized view: %', r.matviewname;
    END LOOP;
END
$$;

-- Transfer ownership of all functions (with proper signature handling)
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT
            p.proname as function_name,
            pg_get_function_identity_arguments(p.oid) as args
        FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'public'
    LOOP
        EXECUTE 'ALTER FUNCTION public.' || quote_ident(r.function_name) || '(' || r.args || ') OWNER TO parobek';
        RAISE NOTICE 'Transferred function: %(%) ', r.function_name, r.args;
    END LOOP;
END
$$;

-- Grant all privileges
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO parobek;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO parobek;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO parobek;
GRANT ALL PRIVILEGES ON SCHEMA public TO parobek;
GRANT USAGE ON SCHEMA public TO parobek;

-- Transfer database ownership
ALTER DATABASE ntsb_aviation OWNER TO parobek;

-- Transfer schema ownership
ALTER SCHEMA public OWNER TO parobek;

\echo ''
\echo '============================================================'
\echo 'Ownership Transfer Complete!'
\echo '============================================================'
\echo ''

-- Show confirmation
\echo 'Tables owned by parobek:'
SELECT tablename, tableowner FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;

\echo ''
\echo 'Database owner:'
SELECT datname, pg_catalog.pg_get_userbyid(datdba) as owner FROM pg_database WHERE datname = 'ntsb_aviation';
