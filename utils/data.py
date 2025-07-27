import pandas as pd
import requests
import zipfile
import io

class FFScraper:
    """A scraper class for downloading and parsing Fama-French factor data from Ken French's data library.
    This class provides methods to download and process both monthly and daily Fama-French factor datasets.
    Attributes:
        base_URL (str): Base URL for the Ken French data library.
        monthly_filename (str): Filename for the monthly factors ZIP file.
        daily_filename (str): Filename for the daily factors ZIP file.
    Args:
        None
    Properties:
        monthly_data (pd.DataFrame): Returns the processed monthly Fama-French factors as a DataFrame indexed by date.
        daily_data (pd.DataFrame): Returns the processed daily Fama-French factors as a DataFrame indexed by date.
    Returns:
        None
    """
    def __init__(self):
        self.base_URL = (
            "https://mba.tuck.dartmouth.edu/"
            "pages/faculty/ken.french/ftp/"
            )
        
        self.monthly_filename = "F-F_Research_Data_Factors_CSV.zip"
        self.daily_filename = "F-F_Research_Data_Factors_daily_CSV.zip"

        self.__monthly_data = None
        self.__daily_data = None

    def _download_csv(self, url):
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.status_code}")

        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        csv_files = [f for f in zip_file.namelist() if f.lower().endswith(".csv")]

        if not csv_files:
            raise Exception("No CSV file found in the ZIP archive.")

        if len(csv_files) > 1:
            print(f"Warning: Multiple CSV files found. Using the first one: {csv_files[0]}.")

        return io.BytesIO(zip_file.open(csv_files[0]).read())

    @property
    def monthly_data(self):
        if self.__monthly_data is None:
            csv_data = self._download_csv(self.base_URL + self.monthly_filename)
            df = pd.read_csv(csv_data, skiprows=3, parse_dates=True).dropna()
            cols = df.columns.str.lower().to_list()
            cols[0] = "date"
            df.columns = cols
            df["date"] = df["date"].str.replace("  ", "")
            df = df[df["date"].str.len()>4].copy()
            df["date"] = pd.to_datetime(df["date"], format="%Y%m")
            self.__monthly_data = df.set_index("date").astype(float)/100
        return self.__monthly_data
    
    @property
    def daily_data(self):
        if self.__daily_data is None:
            csv_data = self._download_csv(self.base_URL + self.daily_filename)
            df = pd.read_csv(csv_data, skiprows=3, parse_dates=True).dropna()
            cols = df.columns.str.lower().to_list()
            cols[0] = "date"
            df.columns = cols
            df["date"] = df["date"].str.replace("  ", "")
            df = df[df["date"].str.len()>7].copy()
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            self.__daily_data = df.set_index("date").astype(float)/100
        return self.__daily_data
    
def read_ff_daily_csv(
    path="data/historical_snapshots/ff_data_daily_202410_snapshot.csv"
):
    csv_f = pd.read_csv(path, skiprows=4, parse_dates=True).dropna().iloc[:, :2]
    csv_f.columns = ['date', 'returns']
    csv_f['date'] = pd.to_datetime(csv_f['date'])
    csv_f = csv_f.set_index('date')['returns'].rename('Unconditional') / 100
    return csv_f