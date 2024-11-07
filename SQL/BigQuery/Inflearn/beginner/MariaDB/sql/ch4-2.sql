-- Active: 1728371458573@@127.0.0.1@3306@inflearn-bigquery
# BigQuery
SELECT
  CAST(1 AS STRING);

# MariaDB
SELECT
  CAST(1 AS CHAR);

SELECT
  CAST(1 AS VARCHAR(255));

# BigQuery
SELECT
  SAFE_CAST('카일스쿨' AS INT64);

# MariaDB
SELECT
  CASE
    WHEN '카일스쿨' REGEXP '^[0-9]+$' 
      THEN CAST('카일스쿨' AS SIGNED)
      ELSE NULL
  END AS result;
