from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.client import GoogleCredentials
import pandas as pd
import datetime
# import time
from numba import jit


@jit
def change_schema(query_response, column_names, result):
    t = 0
    for row in query_response['rows']:
        i = 0
        t += 1
        record = {}
        for field in row['f']:
            if column_names[i]['type'] == 'STRING':
                if 'DATE' in str(column_names[i]['name']).upper():
                    record[column_names[i]['name']] = str(field['v'])[:10]
                    # record[column_names[i]['name']] = datetime.strptime(str(field['v'])[:10],'%Y-%m-%d')
                else:
                    record[column_names[i]['name']] = field['v']
            elif column_names[i]['type'] == 'FLOAT':
                record[column_names[i]['name']] = float(field['v'])
            elif column_names[i]['type'] == 'TIMESTAMP':
                record[column_names[i]['name']] = datetime.datetime.utcfromtimestamp(float(field['v'])).strftime(
                    "%Y-%m-%d %H:%M:%S")
            else:
                record[column_names[i]['name']] = field['v']
            i += 1
        result.append(record)
    return result


def return_bq_query(sql,legacy=True):
    # Grab the application's default credentials from the environment.
    credentials = GoogleCredentials.get_application_default()
    # Construct the service object for interacting with the BigQuery API.
    bigquery_service = build('bigquery', 'v2', credentials=credentials)

    try:
        # [START run_query]
        query_request = bigquery_service.jobs()
        query_data = {
            'useLegacySql': legacy,
            'query': sql
        }

        query_response = query_request.query(projectId='mot-foundations', body=query_data).execute()

        projectID = query_response['jobReference']['projectId']
        jobID = query_response['jobReference']['jobId']
        jobReference = query_response['jobReference']

        result = []
        current_token = None
        next_token = ''
        n_windows = 0
        while next_token != None:

            query_response = query_request.getQueryResults(projectId=projectID, jobId=jobID, pageToken=current_token,
                                                           timeoutMs=50000).execute()
            # print('Waiting....')
            # time.sleep(10)

            if 'totalRows' in query_response:
                total_rows = int(query_response['totalRows'])
            else:
                total_rows = 0

            # print('Total rows returned : %d ' % total_rows)
            #logging.info(query_response)
            if total_rows > 0:

                column_names = []
                # columns = []
                for field in query_response['schema']['fields']:
                    column_names.append(field)
                    # columns.append(field['name'])

                result = change_schema(query_response, column_names, result)
                    # [END setting up result vector]

            n_windows += 1
            if 'pageToken' in query_response:
                next_token = query_response['pageToken']
                # print('Window ' + str(n_windows))
                current_token = next_token
            else:
                next_token = None

        return pd.DataFrame.from_dict(result), total_rows

    except HttpError as err:
        print('Error: {}'.format(err.content))
        raise err