import pandas as pd
import numpy as np


# CSV 로드 및 전처리 함수
def load_and_preprocess_csv(file_path, delimiter='/', header=None, skiprows=5):
    """
    CSV 파일을 읽고 전처리하는 함수
    :param file_path: CSV 파일 경로
    :param delimiter: CSV 파일의 구분자
    :param header: 헤더 정보
    :param skiprows: 건너뛸 행 수
    :return: 전처리된 데이터프레임
    """
    # CSV 파일을 데이터프레임으로 읽기
    df = pd.read_csv(file_path, delimiter=delimiter, header=header, skiprows=skiprows)
    
    # 새로운 열 이름 설정
    new_columns = ['Time', 
                   'Accelerometer', 'Accelerometer x', 'Accelerometer y', 'Accelerometer z',  # 가속도 센서
                   'Magnetometer', 'Magnetometer x', 'Magnetometer y', 'Magnetometer z',      # 지자기 센서
                   'Gyroscope', 'Gyroscope x', 'Gyroscope y', 'Gyroscope z',                 # 자이로 센서
                   'Pressure', 'Pressure2',                                                 # 기압 센서
                   'Oriention', 'Oriention x', 'Oriention y', 'Oriention z',                 # Orientation API [Azimuth, Pitch, Roll]
                   'gravity', 'gravity x', 'gravity y', 'gravity z',                         # 중력 API
                   'no']
    
    # 열 이름 적용
    df.columns = new_columns
    
    # Time 열을 datetime 형식으로 변환
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    
    # 시작 시간을 0으로 설정
    start_time = df['Time'].iloc[0]
    df['Elapsed Time'] = (df['Time'] - start_time).dt.total_seconds()
    
    return df