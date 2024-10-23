-- Active: 1728371458573@@127.0.0.1@3306@inflearn-bigquery

-- DATE
-- DATETIME
-- TIMESTAMP
SELECT
    CURRENT_TIMESTAMP() # 형식 예 : 2024-10-23 14:20:08(현재 위치의 시간대를 알려줌)

# BigQuery
SELECT
    DATETIME(CURRENT_TIMESTAMP(), 'Asia/Seoul') # 현재 시간을 서울 시간대로 변경하는 함수(여기선 필요 없음)

# 서버 타임존 설정, 시스템 타임존 설정
# 서버 타임존이 system이면, 시스템 타임존을 따른 다는 뜻
SELECT @@time_zone, @@system_time_zone

# BigQuery
SELECT
  EXTRACT(DATE FROM DATETIME '2024-01-02 14:00:00') AS date, 
  EXTRACT(YEAR FROM DATETIME '2024-01-02 14:00:00') AS year, 
  EXTRACT(MONTH FROM DATETIME '2024-01-02 14:00:00') AS month, 
  EXTRACT(DAY FROM DATETIME '2024-01-02 14:00:00') AS day, 
  EXTRACT(HOUR FROM DATETIME '2024-01-02 14:00:00') AS hour, 
  EXTRACT(MINUTE FROM DATETIME '2024-01-02 14:00:00') AS minute;

# MariaDB
SELECT
    DATE("2024-01-02 14:00:00") AS date,
    YEAR("2024-01-02 14:00:00") AS year,
    MONTH("2024-01-02 14:00:00") AS month,
    DAY("2024-01-02 14:00:00") AS day,
    HOUR("2024-01-02 14:00:00") AS hour,
    MINUTE("2024-01-02 14:00:00") AS minute

# BigQuery
SELECT
  EXTRACT(DAYOFWEEK FROM DATETIME '2024-01-21 14:00:00') AS day_of_week_sun, 
  EXTRACT(DAYOFWEEK FROM DATETIME '2024-01-22 14:00:00') AS day_of_week_mon, 
  EXTRACT(DAYOFWEEK FROM DATETIME '2024-01-23 14:00:00') AS day_of_week_tues, 
  EXTRACT(DAYOFWEEK FROM DATETIME '2024-01-27 14:00:00') AS day_of_week_sat; 

# MariaDB
SELECT
  DAYOFWEEK('2024-01-21 14:00:00') AS day_of_week_sun,  -- Sunday
  DAYOFWEEK('2024-01-22 14:00:00') AS day_of_week_mon,  -- Monday
  DAYOFWEEK('2024-01-23 14:00:00') AS day_of_week_tues, -- Tuesday
  DAYOFWEEK('2024-01-27 14:00:00') AS day_of_week_sat;  -- Saturday

# BigQuery
SELECT
  DATETIME '2024-03-02 14:42:13' AS original_data

# MariaDB
SELECT
    '2024-03-02 14:42:13' AS original_data # MariaDB에서는 DATETIME 키워드 없이 직접 문자열로 DATETIME 값을 처리해야 합니다.

# BigQuery
SELECT
  DATETIME_TRUNC(DATETIME '2024-03-02 14:42:13', DAY) AS day_trunc,
  DATETIME_TRUNC(DATETIME '2024-03-02 14:42:13', YEAR) AS year_trunc,
  DATETIME_TRUNC(DATETIME '2024-03-02 14:42:13', MONTH) AS month_trunc,
  DATETIME_TRUNC(DATETIME '2024-03-02 14:42:13', HOUR) AS hour_trunc;

# MariaDB
SELECT
  DATE_FORMAT('2024-03-02 14:42:13', '%d') AS day_trunc,    -- 일 단위
  DATE_FORMAT('2024-03-02 14:42:13', '%Y') AS year_trunc,   -- 년 단위
  DATE_FORMAT('2024-03-02 14:42:13', '%m') AS month_trunc,  -- 월 단위
  DATE_FORMAT('2024-03-02 14:42:13', '%H') AS hour_trunc;   -- 시간 단위

SELECT
    '2024-03-02 14:42:13' AS date,
    DATE_FORMAT('2024-03-02 14:42:13', '%Y') AS year_trunc,
    DATE_FORMAT('2024-03-02 14:42:13', '%m') AS month_trunc,
    DATE_FORMAT('2024-03-02 14:42:13', '%d') AS day_trunc,
    DATE_FORMAT('2024-03-02 14:42:13', '%H') AS hour_trunc,
    DATE_FORMAT('2024-03-02 14:42:13', '%i') AS minute_trunc,
    DATE_FORMAT('2024-03-02 14:42:13', '%S') AS second_trunc



-- 마지막 날을 알고 싶은 경우
# BigQuery
SELECT
  LAST_DAY(DATETIME '2024-01-03 15:30:00') AS last_day, # 2024-01-31
  LAST_DAY(DATETIME '2024-01-03 15:30:00', MONTH) AS last_day_month, # 2024-01-31
  LAST_DAY(DATETIME '2024-01-03 15:30:00', WEEK) AS last_day_week, # 2024-01-06
  LAST_DAY(DATETIME '2024-01-03 15:30:00', WEEK(SUNDAY)) AS last_day_week_sun, # 2024-01-06
  LAST_DAY(DATETIME '2024-01-03 15:30:00', WEEK(MONDAY)) AS last_day_week_mon; # 2024-01-07

# MariaDB
SELECT
  LAST_DAY('2024-01-03 15:30:00') AS last_day, # 2024-01-31
  LAST_DAY('2024-01-03 15:30:00') AS last_day_month, # 월의 마지막 날, 2024-01-31
  DATE_ADD(DATE('2024-01-03 15:30:00'), INTERVAL (7 - DAYOFWEEK(DATE('2024-01-03 15:30:00'))) DAY) AS last_day_week, # 2024-01-06
  DATE_ADD(DATE('2024-01-03 15:30:00'), INTERVAL (7 - DAYOFWEEK(DATE('2024-01-03 15:30:00')) + 1) DAY) AS last_day_week_sun, # 2024-01-07
  DATE_ADD(DATE('2024-01-03 15:30:00'), INTERVAL (7 - DAYOFWEEK(DATE('2024-01-03 15:30:00')) + 2) DAY) AS last_day_week_mon; # 2024-01-08
-- bigquery와 다른 결과
-- 해당 날짜가 속한 주의 마지막 날짜를 구하는 함수는 존재하지 않음



-- 두 DATETIME의 차이를 알고 싶은 경우
# BigQuery
SELECT
  DATETIME_DIFF(first_datetime, second_datetime, DAY) AS day_diff1, # 1187
  DATETIME_DIFF(second_datetime, first_datetime, DAY) AS day_diff2, # -1187
  DATETIME_DIFF(first_datetime, second_datetime, MONTH) AS month_diff, # 39
  DATETIME_DIFF(first_datetime, second_datetime, WEEK) AS week_diff # 170
FROM(
  SELECT
    DATETIME '2024-04-02 10:20:00' AS first_datetime,
    DATETIME '2021-01-01 15:30:00' AS second_datetime
  )

# MariaDB
SELECT
  TIMESTAMPDIFF(DAY, first_datetime, second_datetime) AS day_diff1, # -1186
  TIMESTAMPDIFF(DAY, second_datetime, first_datetime) AS day_diff2, # 1186
  TIMESTAMPDIFF(MONTH, first_datetime, second_datetime) AS month_diff, # -39
  TIMESTAMPDIFF(WEEK, first_datetime, second_datetime) AS week_diff # -169
FROM (
  SELECT
    '2024-04-02 10:20:00' AS first_datetime,
    '2021-01-01 15:30:00' AS second_datetime
) AS dates;

# 두 결과의 절대값이 다른 이유(ChatGPT)
    # BigQuery는 월 또는 주 차이를 계산할 때 두 날짜가 포함된 전체 월 또는 주를 기준으로 계산할 수 있습니다. 이로 인해 정확한 차이가 아닌 경과된 전체 월 또는 주 수를 반환할 수 있습니다.
    # MariaDB는 두 날짜 간의 실제 날짜 차이를 기반으로 월 또는 주를 계산합니다.



-- 날짜 및 시간 데이터 타입
-- DATE
-- DATETIME : DATE + TIME. 타임존 정보 X
-- TIMESTAMP : 특정 시점에 도장찍은 값. 타임존 정보O
-- UTC : 국제적인 표준 시간. 한국은 UTC+9
-- Millisecond : 1/1000 초
-- Microsecond : 1/1000 ms



-- 시간 데이터 타입 변환하기
-- millisecond => TIMESTAMP : TIMESTAMP_MILLIS
-- microsecond => TIMESTAMP : TIMESTAMP_MICROS
-- DATETIME
  -- 문자열 => DATETIME : PARSE_DATETIME
  -- DATETIME => 문자열 : FORMAT_DATETIME
  -- 현재 DATETIME : CURRENT_DATETIME
  -- 특정 부분 추출 : EXTRACT
  -- 특정 부분 자르기 : DATETIME_TRUNC
  -- 차이 구하기 : DATETIME_DIFF
 