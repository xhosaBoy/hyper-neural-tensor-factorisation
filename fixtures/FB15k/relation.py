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
    fixtures, _ =  os.path.split(os.path.dirname(__file__))
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
        logger.error(f'Could not insert into {tablename}, {e}')

    connection.commit()
    count = cursor.rowcount
    logger.debug(f'{count} Record inserted successfully into {tablename} table')


def insert_records(records, tablename, connection):
    with connection as con:
        cursor = con.cursor()

        for record in records:
            insert_record(record, tablename, cursor, con)


def get_records(relationfile):

    with open(relationfile, 'r') as relationfile:
        records = []

        for line in relationfile:
            record = {}
            _, relation, _ = line.strip().split('\t')
            logger.debug(f'relation: {relation}')

            record['name'] = relation
            logger.debug(f'record: {record}')
            records.append(record)

        logger.info(f'number of records: {len(records)}')

    return records


def main():
    logger.info('Connecting to database...')
    connection = get_connection('scientist',
                                '*********',
                                '127.0.0.1',
                                '5432',
                                'tensor_factorisation_fb15k')
    logger.info('Successfully conntect to database!')

    tablename = 'relation'
    relationfile = get_path('data/FB15k')
    logger.debug(f'relationfile: {relationfile}')

    logger.info('Getting records...')
    dirname, = list(os.walk(relationfile))
    _, _, filenames = dirname
    experiment = ['train.txt', 'valid.txt', 'test.txt']

    records = []

    for filename in filenames:
        if filename in experiment:
            filename = get_path('data/FB15k', filename)
            logger.debug(f'filename: {filename}')

            logger.info('Extending records...')
            records_new = [{'name': value} for value in set([relation['name'] for relation in get_records(filename)])]
            logger.debug(f'records_train: {records_new}')
            logger.debug(f'number of relations so far: {len(records_new)}')
            records.extend(records_new)
            logger.info('Completed extending records!')

    logger.info('Successfully got records!')

    records = [{'name': value} for value in set([relation['name'] for relation in records])]
    logger.info(f'final number of relations: {len(records)}')

    logger.info('Inserting records...')
    insert_records(records, tablename, connection)
    logger.info('Completed getting records!')


if __name__ == '__main__':
    logger.info('Starting ETL...')
    main()
    logger.info('DONE!')
