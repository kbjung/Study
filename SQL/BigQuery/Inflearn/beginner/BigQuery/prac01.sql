# 날짜 변환 테스트

SELECT
  COUNTIF(catch_date = catch_datetime_kr) AS same,
  COUNTIF(catch_date != catch_datetime_kr) AS diff
FROM(
  SELECT
    catch_date,
    catch_datetime,
    DATE(catch_datetime, 'Asia/Seoul') AS catch_datetime_kr # datetime -> date 날짜만 변환됨
    DATETIME(catch_datetime, 'Asia/Seoul') AS catch_datetime_kr # 날짜 시간 모두 변환됨
  FROM basic.trainer_pokemon
)

SELECT
  DATE_DIFF(first_datetime, second_datetime, DAY) AS day_diff1,
  DATE_DIFF(second_datetime, first_datetime, DAY) AS day_diff2,
  DATE_DIFF(first_datetime, second_datetime, MONTH) AS month_diff,
  DATE_DIFF(first_datetime, second_datetime, WEEK) AS week_diff
FROM(
  SELECT
    DATE '2024-04-02' AS first_datetime,
    DATE '2021-01-01' AS second_datetime
  )



# 날짜 차이 테스트

SELECT
  catch_date,
  catch_datetime_kr,
  DATE_DIFF(catch_datetime_kr, catch_date, DAY) AS diff
FROM(
  SELECT
    catch_date,
    catch_datetime,
    DATE(catch_datetime, 'Asia/Seoul') AS catch_datetime_kr
  FROM basic.trainer_pokemon
  )



# datetime diff test
-- datetime 타입끼리만 계산 됨

SELECT
  catch_date,
  catch_datetime_trans,
  catch_datetime_kr,
  DATETIME_DIFF(catch_datetime_kr, catch_datetime_trans, HOUR) AS diff
  -- DATE_DIFF(catch_datetime_kr, catch_datetime_trans, HOUR) AS diff # 위와 동일한 결과
FROM(
  SELECT
    catch_date,
    catch_datetime,
    DATETIME(catch_datetime) AS catch_datetime_trans,
    DATETIME(catch_datetime, 'Asia/Seoul') AS catch_datetime_kr
  FROM basic.trainer_pokemon
  )



# date -> datetime test
-- date 타입은 시간 정보가 없어서 datetime 타입으로 변환시 시간부분은 00:00:00으로 채워짐

SELECT
  catch_date,
  DATETIME(catch_date) AS dt
FROM basic.trainer_pokemon
