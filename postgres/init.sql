-- PostgreSQL initialization script
-- Runs automatically when the postgres container starts for the first time

-- Create the database (already handled by POSTGRES_DB env var, but kept for clarity)
-- Ensure the coupon_user has all privileges

GRANT ALL PRIVILEGES ON DATABASE coupon_db TO coupon_user;

-- Set timezone
SET timezone = 'UTC';

-- Optional: create indexes that SQLAlchemy won't auto-create
-- (SQLAlchemy handles table creation, but we can add composite indexes here)

-- These will be created AFTER SQLAlchemy initializes the tables via a separate migration
-- The services call init_db() on startup which runs Base.metadata.create_all()
