-- Drop table if it already exists
DROP TABLE IF EXISTS World;

-- Create World table
CREATE TABLE World (
    name VARCHAR(100) PRIMARY KEY,
    continent VARCHAR(50),
    area INT,
    population INT,
    gdp BIGINT
);

-- Insert data into World table
INSERT INTO World (name, continent, area, population, gdp) VALUES
('Afghanistan', 'Asia', 652230, 25500100, 20343000000),
('Albania', 'Europe', 28748, 2831741, 12960000000),
('Algeria', 'Africa', 2381741, 37100000, 188681000000),
('Andorra', 'Europe', 468, 78115, 3712000000),
('Angola', 'Africa', 1246700, 20609294, 100990000000);
