import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import base64

def convert_csv_to_parquet(input_file_path, output_file_path, drop_option):
    df = pd.read_csv(input_file_path)

    if drop_option == 'row':
        df = df.dropna()
    elif drop_option == 'column':
        df = df.dropna(axis=1)

    df['before'] = df['before'].apply(lambda x: base64.b64encode(open(x, 'rb').read()).decode())
    df['after'] = df['after'].apply(lambda x: base64.b64encode(open(x, 'rb').read()).decode())

    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file_path, compression='zstd')
    table = pq.read_table(output_file_path)
    df = table.to_pandas()

convert_csv_to_parquet('output.csv', 'data.parquet', 'row')