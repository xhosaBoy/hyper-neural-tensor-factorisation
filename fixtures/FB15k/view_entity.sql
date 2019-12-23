drop view if exists entity;

create view entity as
select subject as synset_id, name
from train left join entity_freebase
on subject = synset_id
union
select subject, name
from valid left join entity_freebase
on subject = synset_id
union
select subject, name
from test left join entity_freebase
on subject = synset_id
union
select object, name
from train left join entity_freebase
on object = synset_id
union
select object, name
from valid left join entity_freebase
on object = synset_id
union
select object, name
from test left join entity_freebase
on object = synset_id;
