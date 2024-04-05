# how to group multiple tasks (various downloads and transforms)
# to single group DAG

from airflow import DAG
from airflow.operators.python import PythonOperator, get_current_context
import requests

# Now they are called Task Groups and Subdags are deprecated
# but still work
# from airflow.operators.subdag import SubDagOperator # important to implement grouping

from datetime import datetime

# request headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
    'Accept-Encoding': 'utf-8'
}

# pulling up location IDs
def recurrent_area_search(areas: list):
    id_area = []
    for area in areas:
        id_area.append((area['id'], area['name']))
        if 'areas' in area.keys():
            if len(area['areas']) > 0:
                id_area.extend(recurrent_area_search(area['areas']))
    return id_area
def pull_locations(ti):
    area_codes = requests.get('https://api.hh.ru/areas').json()
    ti.xcom_push(
        key="area_codes", 
        value={a.lower(): i for i, a in recurrent_area_search(area_codes)}
    )

def print_locations(ti):
    print(
        'AREA CODES:', ti.xcom_pull(key="area_codes")
    )

with DAG('data_prep', start_date=datetime(2024, 3, 24), 
    schedule_interval='@weekly', catchup=False,
    render_template_as_native_obj=True) as dag:
 
    args = {
        'start_date': dag.start_date, 
        'schedule_interval': dag.schedule_interval,
        'catchup': dag.catchup
    }

    locPull = PythonOperator(
        task_id='pull_locations',
        python_callable=pull_locations
    )
 
    locPrint = PythonOperator(
        task_id='print_locations',
        python_callable=print_locations
    )

    # [download_a, download_b, download_c] >> check_files >> [transform_a, transform_b, transform_c]
    # the first group transforms to simply "downloads"
    locPull >> locPrint