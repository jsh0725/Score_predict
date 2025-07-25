#CSV 로딩 및 시즌 분할(2020~2023 -> train / 2024~2025 -> test)
import pandas as pd
import re

def load_and_split_by_season(csv_path: str, train_seasons: list, test_seasons: list):
    df = pd.read_csv(csv_path)

    #Season이 숫자형이 아닐 경우 정수로 반환하도록 처리
    df['Season'] = df['Season'].astype(int)

    #train / test split
    train_df = df[df['Season'].isin(train_seasons)].reset_index(drop = True)
    test_df = df[df['Season'].isin(test_seasons)].reset_index(drop = True)

    return train_df, test_df

#Result(경기 결과)를 보면 L / D / W의 3가지 종류(다중)으로 되어 있음을 알 수 있음
#이를 0, 1, 2로 다중 분류용 숫자로 변환
def encode_number(df: pd.DataFrame) -> pd.DataFrame:
    label_map = {'L': 0, 'D': 1, 'W': 2}
    df['Label'] = df['Result'].map(label_map)
    return df

#여러 열들 중 문자형으로 저장되어 있는 열 존재(Day, Round, Venue 등)
#이를 정수형으로 인코딩
def encode_chracter(df: pd.DataFrame) -> pd.DataFrame:
    df['Venue'] = df['Venue'].map({'Home': 1, 'Away': 0})

    day = {'Mon':0, 'Tue':1, 'Wed':2, 'Thu':3, 'Fri':4, 'Sat':5, 'Sun': 6}
    df['Day'] = df['Day'].map(day)

    def extract_round(x):
        if pd.isnull(x):
            return None
        match = re.findall(r'\d+', str(x))
        return int(match[0]) if match else None

    df['Round'] = df['Round'].apply(extract_round)

    return df

#Data 열을 활용하여 피처 생성, 모델 입력에 활용(MatchMonth, MatchDayOfWeek, MatchDay, MatchOrdinal)

def process_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce')

    #파생 변수 생성
    df['MatchMonth'] = df['Date'].dt.month
    df['MatchDayOfWeek'] = df['Date'].dt.weekday
    df['MatchDay'] = df['Date'].dt.day

    #날짜 기준 정렬 후 순번 부여
    df = df.sort_values('Date').reset_index(drop = True)
    df['MatchOrdinal'] = range(1, len(df) + 1)

    return df

#불필요한 열 제거 및 결측값 대체
def remove_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = ['Unnamed: 0', 'Match Report', 'Notes']
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors = 'ignore')

    #NaN -> 0
    num_columns = df.select_dtypes(include = ['float64', 'int64']).columns
    df[num_columns] = df[num_columns].fillna(0)

    return df
 
 #입력(X), 타겟(y) 분리

def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df['Label']

    #제거할 열
    exclude_columns = ['Label', 'Result', 'Team', 'Opponent', 'Date', 'Time', 'Comp']
    X = df.drop(columns = [col for col in exclude_columns if col in df.columns], errors='ignore')

    #문자형 열 제거
    X = X.select_dtypes(include = ['number'])

    return X, y