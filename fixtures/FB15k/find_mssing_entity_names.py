# std
import os
import sys
import json
import urllib.parse
import urllib.request
import logging

# 3rd party
import psycopg2
from psycopg2.extras import Json
from psycopg2.extensions import AsIs


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_path(filename, dirname=None):
    root, _ = os.path.split(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(root, dirname, filename) if dirname else os.path.join(root, filename)
    logger.debug(f'path: {path}')
    return path


def get_connection(user, password, host, port, database):
    """Database connection getter.
        Args:
            user (str): database user.
            password (str): database user password.
            host (str): database IP.
            port (str): database port.
            database (str): database name.
        Returns:
            connection (obj): postgres database connection.
        """
    connection = psycopg2.connect(user=user,
                                  password=password,
                                  host=host,
                                  port=port,
                                  database=database)
    return connection


def get_entity_ids(tablename, connection):
    with connection as con:
        cursor = con.cursor()
        query = 'SELECT synset_id FROM %s WHERE name is null'
        logger.debug(f"cursor.mogrify: {cursor.mogrify(query, (AsIs(tablename),))}")

        try:
            cursor.execute(query, (AsIs(tablename),))
        except Exception as e:
            logger.error(f'Could select from {tablename}, {e}')

        entity_ids = cursor.fetchall()
        logger.debug(f'entity_ids: {entity_ids}')

    return entity_ids


def fetch_missing_names(synset_ids):
    path = get_path('google_kg.txt', 'credentials')
    api_key = open(path).read()
    ids = '&ids='.join([item[0] for item in synset_ids])
    print(f'ids: {ids}')
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'ids': ids,
        'limit': 1000,
        'indent': True,
        'key': api_key,
    }

    url = service_url + '?' + urllib.parse.urlencode(params, safe='=/&')
    logger.debug(f'url: {url}')

    response = json.loads(urllib.request.urlopen(url).read())
    logger.debug(f'response: {response}')

    entities = list(response['itemListElement'])
    entity_with_names = {}

    for index, entity in enumerate(entities):
        _, id = entity['result']['@id'].split(':')
        logger.debug(f'entity id: {id}')
        entity_with_names[id] = entity['result']['name']
        logger.debug(f"entity {index + 1} - {entity['result']['name'] + ' (' + str(entity['resultScore']) + ')'}")

    logger.debug(f'entity_with_names: {entity_with_names}')

    return entity_with_names


def insert_record(record, tablename, cursor, connection):

    columns = record.keys()
    logger.debug(f'columns: {columns}')
    values = record.values()
    logger.debug(f'values: {values}')
    values = list(map(lambda x: Json(x) if isinstance(x, dict) else x, values))

    insert_statement = 'INSERT INTO %s (%s) VALUES %s'
    logger.debug(f"cursor.mogrify: {cursor.mogrify(insert_statement, (AsIs(tablename), AsIs(','.join(columns)), tuple(values)))}")

    try:
        cursor.execute(insert_statement, (AsIs(tablename), AsIs(','.join(columns)), tuple(values)))
    except Exception as e:
        logger.error(f'Could not insert into {tablename}, {e}')

    connection.commit()
    count = cursor.rowcount
    logger.debug(f'{count} Record inserted successfully into {tablename} table')


def insert_records(entities_with_names, tablename, connection):
    with connection as con:
        cursor = con.cursor()

        for synset_id, name in entities_with_names.items():
            record = {}
            record['synset_id'] = synset_id
            record['name'] = name
            insert_record(record, tablename, cursor, connection)

def main():

    logger.info('Connecting to database...')
    connection = get_connection('scientist',
                                '*********',
                                '127.0.0.1',
                                '5432',
                                'tensor_factorisation_fb15k')
    logger.info('Successfully conntect to database!')

    tablename = 'entity'

    logger.info('Finding missing entity names...')
    entity_ids = get_entity_ids(tablename, connection)
    logger.info(f'Found {len(entity_ids)} missing entity names!')

    logger.info(f'Fetching missing names...')
    entities_with_names = fetch_missing_names(entity_ids)
    logger.info(f'Fetched {len(entities_with_names)} missing entity names!')

    logger.info('Inserting records...')
    tablename = 'entity_freebase'
    insert_records(entities_with_names, tablename, connection)
    logger.info('Successfully inserted records!')


if __name__ == '__main__':
    logger.info('Populating missing entity names...')
    main()
    logger.info('DONE!')
