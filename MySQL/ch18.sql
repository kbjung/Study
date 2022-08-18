-- 스토어드 프로시저
-- SQL에 프로그래밍기능을 추가한 것.

/* 
스토어드 프로시저 만들기 문법
DELIMITER $$
CREATE PROCEDURE 스토어드_프로시저_이름(IN 또는 OUT 매개변수)
BEGIN
    SQL 프로그래밍 코드
END $$
DELIMITER ;
*/

-- 스토어드 프로시저 실행 문법
-- CALL 스토어드_프로시저_이름();

use market_db;

drop procedure if exists user_proc;
delimiter //
create procedure user_proc()
begin
	select * from member;
end //
delimiter ;

call user_proc();

drop procedure user_proc;

drop procedure if exists user_proc1;
delimiter //
create procedure user_proc1(in userName varchar(10))
begin
	select * from member where mem_name = userName;
end //
delimiter ;

call user_proc1('에이핑크');


drop procedure if exists user_proc2;
delimiter //
create procedure user_proc2(
	in userNumber int, 
    in userHeight int
    )
begin
	select * from member 
		where mem_number > userNumber and height > userHeight;
end //
delimiter ;

call user_proc2(6, 165)


drop procedure if exists user_proc3;

delimiter //
create procedure user_proc3(
	in txtValue char(10),
    out outValue int) -- outvalue 결과값을 출력
begin
	insert into noTable values(null, txtValue);
    select max(id) into outValue from noTable; -- max(id) 값을 outValue로 보낸다.
end //
delimiter ;

desc noTable;

create table if not exists noTable(
	id int auto_increment primary key,
	txt char(10)
);

call user_proc3('테스트1', @myValue); -- 반복실행하면 결과 값이 증가한다. 인덱스가 자동으로 증가하도록 설정했으므로.
select @myValue;

drop procedure if exists ifelse_proc;
delimiter //
create procedure ifelse_proc(
	in memName varchar(10)
    )
begin
	declare debutYear int;
    select year(debut_date) into debutYear from member
		where mem_name = memName;
	if debutYear >= 2015 then
		select '신인 가수네요. 화이팅 하세요.' as '메시지';
	else
		select '고참 가수네요. 그동안 수고하셨어요.' as '메시지';
	end if;
end //
delimiter ;

call ifelse_proc('소녀시대');


drop procedure if exists while_proc;
delimiter //
create procedure while_proc()
begin
	declare hap int;
    declare num int;
    set hap = 0;
    set num = 1;
    
    while num <= 100 do
		set hap = hap + num;
        set num = num + 1;
	end while;
    select hap as '1 ~ 100 합계';
end //
delimiter ;

call while_proc();


-- 동적 SQL

drop procedure if exists dynamic_proc;
delimiter //
create procedure dynamic_proc(
	in tableName varchar(20))
begin
	set @sqlQuery = concat('select * from ', tableName);
    prepare myQuery from @sqlQuery;
    execute myQuery;
    deallocate prepare myQuery;
end //
delimiter ;

call dynamic_proc('member');
call dynamic_proc('buy');