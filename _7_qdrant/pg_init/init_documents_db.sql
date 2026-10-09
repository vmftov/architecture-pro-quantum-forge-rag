CREATE DATABASE documents_db;

\c metadata_db;

CREATE TABLE documents (
    sensor_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(64) NOT NULL,
    location VARCHAR(255) NOT NULL,
    unit VARCHAR(16),
    status VARCHAR(16),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE chunks (
    relay_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(64) NOT NULL,
    location VARCHAR(255) NOT NULL,
    unit VARCHAR(16),
    status VARCHAR(16),
    desired_value DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sensors_name ON sensors(name);
CREATE INDEX idx_sensors_type ON sensors(type);
CREATE INDEX idx_sensors_location ON sensors(location);
CREATE INDEX idx_sensors_status ON sensors(status);

CREATE INDEX idx_relays_name ON relays(name);
CREATE INDEX idx_relays_type ON relays(type);
CREATE INDEX idx_relays_location ON relays(location);
CREATE INDEX idx_relays_status ON relays(status);
