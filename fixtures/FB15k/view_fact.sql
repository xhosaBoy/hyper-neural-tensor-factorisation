drop view if exists fact_train;
drop view if exists fact_valid;
drop view if exists fact_test;

create view fact_train (id, subject, s_name, predicate, object, o_name) as
select S.id, S.subject, S.name, S.predicate, O.object, O.name
from (select *
from train left join entity
on subject = synset_id) S, (select *
from train left join entity
on object = synset_id) O
where S.id = O.id;

create view fact_valid (id, subject, s_name, predicate, object, o_name) as
select S.id, S.subject, S.name, S.predicate, O.object, O.name
from (select *
from valid left join entity
on subject = synset_id) S, (select *
from valid left join entity
on object = synset_id) O
where S.id = O.id;

create view fact_test (id, subject, s_name, predicate, object, o_name) as
select S.id, S.subject, S.name, S.predicate, O.object, O.name
from (select *
from test left join entity
on subject = synset_id) S, (select *
from test left join entity
on object = synset_id) O
where S.id = O.id;
