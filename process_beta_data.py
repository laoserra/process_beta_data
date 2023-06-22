#!/usr/bin/env python3

import sys
import pandas as pd

classes_tf2 = ['car', 'person', 'bicycle', 'motorcycle', 'bus', 'truck']
classes_yolo = ['car', 'pedestrian', 'cyclist',
                'motorcycle', 'bus', 'lorry', 'van', 'taxi']


def clean_data(df_raw):
    # remove duplicates 
    df_raw.drop_duplicates(inplace=True)
    # Eliminate the "Parking" preset for cam A13. Not needed.
    df_raw = df_raw[~(df_raw['camera_ref'].str.contains('Parking'))].copy()
    df = pd.DataFrame()
    df['image_proc'] = pd.to_datetime(df_raw['image_proc'], format="%Y-%m-%d %H:%M:%S%z")
    df['image_proc'] = df['image_proc'].dt.tz_convert('Europe/London')
    df['image_capt'] = pd.to_datetime(df_raw['image_capt'], format="%Y-%m-%d %H:%M:%S%z")
    df['image_capt'] = df['image_capt'].dt.tz_convert('Europe/London')
    df['camera_ref'] = df_raw['camera_ref']
    df['warnings'] = df_raw['warnings']
    df['class_name'] = df_raw['class_name']

    return df


def group_by_class_name(df, model):
    if model == 'yolo':
        classes = classes_yolo
    else:
        classes = classes_tf2
        model = 'tf2'
    # filter data where detections are absent
    nan = 0
    if df.isna().any().any():
        df_nan = df[df.isna().any(axis=1)]
        # drop columns 'score' and 'class_name'
        df_nan = df_nan[['image_proc', 'image_capt', 'camera_ref', 'warnings']]
        nan = 1
    # group detections by image attributes and class object
    df_agg = df.groupby(['image_proc',
                         'image_capt',
                         'camera_ref',
                         'warnings',
                         'class_name'])
    # transpose class_name to columns with counts
    df_agg = df_agg.size().unstack(fill_value=0).reset_index()
    # check for missing classes of objects and update df_agg
    df_agg_classes = df_agg.columns[4:]
    if len(df_agg_classes) < len(classes):
        missing_classes = list(set(classes) - set(df_agg_classes))
        for col in missing_classes:
            df_agg[col] = 0
    # concatenate df_nan with df_agg if df_nan existent
    if nan:
        df = pd.concat([df_nan, df_agg])
        for col in classes:
            df[col] = df[col].fillna(0).astype(int)
    else:
        df = df_agg
    # reorder columns
    columns_order = ['image_proc', 'image_capt', 'camera_ref'] \
                    + classes \
                    + ['warnings']
    df = df[columns_order]
    # insert name of model
    df.insert(3, 'model_name', model)
    # sum all numeric columns (class names and warnings). 
    # warnings are all zero...if not there will be no detections
    df['temp'] = df.sum(axis=1, numeric_only=True)
    # remove duplicated rows by image capture, camera ref and temp
    df = df.drop_duplicates(subset=['image_capt', 'camera_ref', 'temp'])
    df = df[~(df.duplicated(subset=['image_capt', 'camera_ref'], keep=False) & df['temp'].eq(0))]
    df = df.sort_values(by=['image_proc', 'image_capt', 'camera_ref']) 
    # remove  column 'temp'
    df = df.iloc[:,:-1]
    # copy all A36 rows to file for latter processing
    #df_A36 = df.loc[df['camera_ref'] == 'A36',:]
    #df_A36.to_csv(f'beta-stage-A36-data-20230510-20230511-{model}.csv')
    # Eliminate rows with camera_ref==A36 
    # because this camera ref is duplicated per each 30min interval.
    #df = df.loc[~(df['camera_ref'] == 'A36'),:]
    # select duplicated rows by image capture and camera ref. Remove the row where temp=0

    return df, model


def main(file_path):
    data = pd.read_csv(file_path)
    df_clean = clean_data(data)
    df, model = group_by_class_name(df_clean, file_path[:4])
    #report = f'cctv-report-v2-{model}-beta-20230509.csv'
    #df.to_csv(report, index=False)
    #sys.exit()
    # group by day of capture and write it to gzip
    # (the values passed to Grouper take precedence)
    grouped = df.groupby(pd.Grouper(key="image_capt", freq="D"))
    for date, group in grouped:
        day = date.strftime(format='%Y%m%d')
        report = f'./daily_reports/{model}/cctv-report-v2-{model}-{day}.csv.gz'
        group.to_csv(report, index=False, compression='gzip')


if __name__ == '__main__':
    main(sys.argv[1])
