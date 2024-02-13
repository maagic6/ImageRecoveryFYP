import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
from datasets import Features, Image, Value

features = Features({"before": Image(), "instruction": Value("string"), "after": Image()})
schema = features.arrow_schema

def convert_csv_to_parquet(input_file_path, output_file_path, drop_option, schema):
    df = pd.read_csv(input_file_path)

    if drop_option == 'row':
        df = df.dropna()
    elif drop_option == 'column':
        df = df.dropna(axis=1)

    df['before'] = df['before'].apply(lambda x: {'bytes': open(x, 'rb').read()})
    df['after'] = df['after'].apply(lambda x: {'bytes': open(x, 'rb').read()})

    table = pa.Table.from_pandas(df, schema=schema)
    pq.write_table(table, output_file_path, compression='snappy')
    table = pq.read_table(output_file_path)
    df = table.to_pandas()
    print(df.head())

convert_csv_to_parquet('data.csv', 'data.parquet', 'row', schema)