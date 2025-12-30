-- Reset all ragd data (keeps schema + extension).
TRUNCATE TABLE chunks, documents, collections RESTART IDENTITY CASCADE;

-- Optional: also reset API keys.
-- TRUNCATE TABLE api_keys RESTART IDENTITY CASCADE;
