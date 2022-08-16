USE market_db;

SELECT mem_id, mem_name, debut_date 
FROM member 
ORDER BY debut_date;

SELECT mem_id, mem_name, debut_date 
FROM member 
ORDER BY debut_date ASC; -- 기본값 ASC, 오름차순

SELECT mem_id, mem_name, debut_date 
FROM member 
ORDER BY debut_date DESC; -- 내림차순

SELECT mem_id, mem_name, debut_date, height
FROM member
ORDER BY height DESC
WHERE height >= 164; -- 오류발생

SELECT mem_id, mem_name, debut_date, height
FROM member
WHERE height >= 164
ORDER BY height DESC; -- WHERE, ORDER BY 순으로 작성해야 한다.

SELECT mem_id, mem_name, debut_date, height
FROM member
WHERE height >= 164
ORDER BY height DESC, debut_date ASC; -- 먼저 height순으로 내림차순, height가 같다면 debut_date로 오름차순 정렬한다.

SELECT * FROM member LIMIT 3;

SELECT mem_name, debut_date
FROM member
ORDER BY debut_date
LIMIT 3;

SELECT mem_name, height
FROM member
ORDER BY height DESC
LIMIT 3, 2; -- 3번째부터 2개만 출력(3, 4등), 2등이 동일한 2개가 존재(잇지, 트와이스) 따라서 3등은 4번째인 여자친구.

SELECT addr FROM member;

SELECT DISTINCT addr FROM member; -- 중복 제거해서 하나만 출력

SELECT mem_id, amount FROM buy ORDER BY mem_id;

SELECT mem_id, SUM(amount) FROM buy GROUP BY mem_id;

SELECT mem_id "회원 아이디", SUM(price*amount) '총 구매 금액' 
FROM buy GROUP BY mem_id;

SELECT AVG(amount) FROM buy; -- 전체 구매 평균 개수

SELECT mem_id, AVG(amount) -- 멤버 id 별 구매 평균 개수
FROM buy
GROUP BY mem_id;

SELECT COUNT(*) FROM member; -- 전체 행의 개수

SELECT COUNT(phone1) '연락처가 있는 회원' FROM member;

SELECT mem_id '회원 아이디', SUM(price*amount) '총 구매 금액'
FROM buy
WHERE SUM(price*amount) > 1000 -- 오류 발생. group by에 where 사용 못함.
GROUP BY mem_id;

SELECT mem_id '회원 아이디', SUM(price*amount) '총 구매 금액'
FROM buy
GROUP BY mem_id
HAVING SUM(price*amount) > 1000;

SELECT mem_id '회원 아이디', SUM(price*amount) '총 구매 금액'
FROM buy
GROUP BY mem_id
HAVING SUM(price*amount) > 1000
ORDER BY SUM(price*amount) DESC;