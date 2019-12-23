# std
import os
import sys
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


def get_path(dirname, filename=None):
    """File path getter.
        Args:
            dirname (str): File directory.
            filename (str): File name.
        Returns:
            filepath (str): Full file file path.
        """
    fixtures, _ = os.path.split(os.path.dirname(__file__))
    project = os.path.dirname(fixtures)
    path = os.path.join(project, dirname, filename) if filename else os.path.join(project, dirname)

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
        logger.debug(f'Could not insert into {tablename}, {e}')

    connection.commit()
    count = cursor.rowcount
    logger.debug(f'{count} Record inserted successfully into {tablename} table')


def insert_records(entityfile, tablename, connection):

    with open(entityfile, 'r', encoding='utf-8') as entityfile:

        with connection as con:
            cursor = con.cursor()

            for index, line in enumerate(entityfile):
                record = {}
                entity = line.strip().split('\t')
                if len(entity) > 1:
                    synset_id, intelligible_name = entity
                    logger.debug(f'synset_id: {synset_id}, intelligible_name: {intelligible_name}')
                else:
                    entity.append('')
                    synset_id, intelligible_name = entity
                    logger.debug(f'synset_id: {synset_id}, intelligible_name: {intelligible_name}')

                name = intelligible_name
                logger.debug(f'name: {name}')

                record['synset_id'] = synset_id
                record['name'] = name
                logger.debug(f'record: {record}')

                insert_record(record, tablename, cursor, con)

                if index % 100000 == 0:
                    logger.info(f'{index + 1} lines processed')


def main():
    logger.info('Connecting to database...')
    connection = get_connection('scientist',
                                '*********',
                                '127.0.0.1',
                                '5432',
                                'tensor_factorisation_fb15k')
    logger.info('Successfully conntect to database!')

    tablename = 'entity_freebase'
    entityfile = get_path('data/FB15k', 'mid2name.tsv')
    logger.debug(f'entityfile: {entityfile}')

    logger.info('Inserting records...')
    insert_records(entityfile, tablename, connection)
    logger.info('Successfully inserted records!')


if __name__ == '__main__':
    logger.info('Starting ETL...')
    main()
    logger.info('DONE!')
