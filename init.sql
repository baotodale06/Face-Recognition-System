CREATE TABLE IF NOT EXISTS attendance_log(
    id  SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    similarity REAL,
    image_path VARCHAR(500),
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS unknown_log(
    id SERIAL PRIMARY KEY,
    image_path VARCHAR(500),
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    note TEXT
);

CREATE TABLE IF NOT EXISTS app_settings(
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL
);

--- default settings
INSERT INTO app_settings (key, value) VALUES
    ('det_weight', './weights/det_10g.onnx'),
    ('rec_weight', './weights/w600k_mbf.onnx'),
    ('confidence_thresh', '0.5'),
    ('similarity_thresh', '0.4'),
    ('unknown_debounce_sec', '5'),
    ('known_debounce_min', '1')
ON CONFLICT (key) DO NOTHING;


-- Indexing 
CREATE INDEX IF NOT EXISTS idx_attendance_name on attendance_log (name);
CREATE INDEX IF NOT EXISTS idx_attendance_time on attendance_log (detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_unknown_time on unknown_log (detected_at DESC);