-- DATE
-- DATETIME
-- TIMESTAMP

-- SELECT
--   CURRENT_TIMESTAMP() AS timestamp_col,
--   DATETIME(CURRENT_TIMESTAMP(), 'Asia/Seoul') AS datetime_col;

-- SELECT
--   EXTRACT(DATE FROM DATETIME '2024-01-02 14:00:00') AS date, 
--   EXTRACT(YEAR FROM DATETIME '2024-01-02 14:00:00') AS year, 
--   EXTRACT(MONTH FROM DATETIME '2024-01-02 14:00:00') AS month, 
--   EXTRACT(DAY FROM DATETIME '2024-01-02 14:00:00') AS day, 
--   EXTRACT(HOUR FROM DATETIME '2024-01-02 14:00:00') AS hour, 
--   EXTRACT(MINUTE FROM DATETIME '2024-01-02 14:00:00') AS minute;

-- SELECT
--   EXTRACT(DAYOFWEEK FROM DATETIME '2024-01-21 14:00:00') AS day_of_week_sun, 
--   EXTRACT(DAYOFWEEK FROM DATETIME '2024-01-22 14:00:00') AS day_of_week_mon, 
--   EXTRACT(DAYOFWEEK FROM DATETIME '2024-01-23 14:00:00') AS day_of_week_tues, 
--   EXTRACT(DAYOFWEEK FROM DATETIME '2024-01-27 14:00:00') AS day_of_week_sat; 

-- SELECT
--   DATETIME '2024-03-02 14:42:13' AS original_data,
--   DATETIME_TRUNC(DATETIME '2024-03-02 14:42:13', DAY) AS day_trunc,
--   DATETIME_TRUNC(DATETIME '2024-03-02 14:42:13', YEAR) AS year_trunc,
--   DATETIME_TRUNC(DATETIME '2024-03-02 14:42:13', MONTH) AS month_trunc,
--   DATETIME_TRUNC(DATETIME '2024-03-02 14:42:13', HOUR) AS hour_trunc;



-- 문자열로 저장된 DATETIME을 DATETIME 타입으로 바꾸는 함수
-- 문자열 => DATETIME : PARSE_DATETIME
-- SELECT
--   PARSE_DATETIME('%Y-%m-%d %H:%M:%S', '2024-01-11 12:35:35') AS parse_datetime;



-- DATETIME 타입의 데이터를 특정 형태의 문자열 데이터로 변환하는 함수
-- DATETIME => 문자열 : FORMAT_DATETIME

-- SELECT
--   FORMAT_DATETIME('%c', DATETIME '2024-01-11 12:35:35') AS formatted;


-- 마지막 날을 알고 싶은 경우
-- SELECT
--   LAST_DAY(DATETIME '2024-01-03 15:30:00') AS last_day,
--   LAST_DAY(DATETIME '2024-01-03 15:30:00', MONTH) AS last_day_month,
--   LAST_DAY(DATETIME '2024-01-03 15:30:00', WEEK) AS last_day_week,
--   LAST_DAY(DATETIME '2024-01-03 15:30:00', WEEK(SUNDAY)) AS last_day_week_sun,
--   LAST_DAY(DATETIME '2024-01-03 15:30:00', WEEK(MONDAY)) AS last_day_week_mon;



-- 두 DATETIME의 차이를 알고 싶은 경우
SELECT
  DATETIME_DIFF(first_datetime, second_datetime, DAY) AS day_diff1,
  DATETIME_DIFF(second_datetime, first_datetime, DAY) AS day_diff2,
  DATETIME_DIFF(first_datetime, second_datetime, MONTH) AS month_diff,
  DATETIME_DIFF(first_datetime, second_datetime, WEEK) AS week_diff
FROM(
  SELECT
    DATETIME '2024-04-02 10:20:00' AS first_datetime,
    DATETIME '2021-01-01 15:30:00' AS second_datetime
  )



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
 