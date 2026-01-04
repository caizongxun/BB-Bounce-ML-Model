"""
從 Hugging Face 下載加密貨幣 OHLCV 資料的腳本

資料結構：
根目錄: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/tree/main
klines 資料夾: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/tree/main/klines

幣種結構（以BTC為例）:
https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/tree/main/klines/BTCUSDT
├── BTC_15m.parquet  (15分鐘K線)
└── BTC_1h.parquet   (1小時K線)
"""

import pandas as pd
from huggingface_hub import hf_hub_download
import os
from pathlib import Path
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HF 數據集信息
HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
HF_REPO_TYPE = "dataset"

# 支援的幣種列表及其檔案名稱對應
SYMBOLS = {
    "BTCUSDT": {"short_name": "BTC", "timeframes": ["15m", "1h"]},
    "ETHUSDT": {"short_name": "ETH", "timeframes": ["15m", "1h"]},
    "BNBUSDT": {"short_name": "BNB", "timeframes": ["15m", "1h"]},
    "XRPUSDT": {"short_name": "XRP", "timeframes": ["15m", "1h"]},
    "SOLUSDT": {"short_name": "SOL", "timeframes": ["15m", "1h"]},
    "ADAUSDT": {"short_name": "ADA", "timeframes": ["15m", "1h"]},
    "DOGEUSDT": {"short_name": "DOGE", "timeframes": ["15m", "1h"]},
    "MATICUSDT": {"short_name": "MATIC", "timeframes": ["15m", "1h"]},
    "LINKUSDT": {"short_name": "LINK", "timeframes": ["15m", "1h"]},
    "LTCUSDT": {"short_name": "LTC", "timeframes": ["15m", "1h"]},
    "UNIUSDT": {"short_name": "UNI", "timeframes": ["15m", "1h"]},
    "FILUSDT": {"short_name": "FIL", "timeframes": ["15m", "1h"]},
    "OPUSDT": {"short_name": "OP", "timeframes": ["15m", "1h"]},
    "NEARUSDT": {"short_name": "NEAR", "timeframes": ["15m", "1h"]},
}


def download_kline_data(symbol: str, timeframe: str, local_cache_dir: str = "data_cache") -> pd.DataFrame:
    """
    從 HF 下載指定幣種和時間框架的 K 線資料
    
    參數：
    -----------
    symbol : str
        幣種代碼，例如 "BTCUSDT"
    timeframe : str
        時間框架，例如 "15m" 或 "1h"
    local_cache_dir : str
        本地快取目錄
        
    返回：
    -----------
    pd.DataFrame
        包含 OHLCV 資料的 DataFrame
    """
    
    try:
        # 確保快取目錄存在
        Path(local_cache_dir).mkdir(parents=True, exist_ok=True)
        
        if symbol not in SYMBOLS:
            logger.error(f"不支援的幣種: {symbol}")
            logger.info(f"支援的幣種: {', '.join(SYMBOLS.keys())}")
            return None
        
        symbol_info = SYMBOLS[symbol]
        short_name = symbol_info["short_name"]
        
        # 構建檔案路徑（在 klines 資料夾內）
        # 例如: klines/BTCUSDT/BTC_15m.parquet
        file_name = f"{short_name}_{timeframe}.parquet"
        file_path = f"klines/{symbol}/{file_name}"
        
        logger.info(f"正在下載 {symbol} {timeframe} 數據...")
        logger.info(f"HF 檔案路徑: {file_path}")
        
        # 從 HF 下載檔案
        local_file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=file_path,
            repo_type=HF_REPO_TYPE,
            cache_dir=local_cache_dir,
            force_download=False,  # 使用快取，除非檔案已刪除
            resume_download=True
        )
        
        # 讀取 parquet 檔案
        df = pd.read_parquet(local_file_path)
        
        # 確保時間戳格式正確
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ 成功下載: {len(df)} 行")
        logger.info(f"時間範圍: {df.index.min()} ~ {df.index.max()}")
        logger.info(f"列: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        logger.error(f"❌ 下載失敗: {str(e)}")
        return None


def download_multiple_symbols(
    symbols: list = None,
    timeframes: list = None,
    local_cache_dir: str = "data_cache"
) -> dict:
    """
    批量下載多個幣種和時間框架的資料
    
    參數：
    -----------
    symbols : list
        幣種列表，預設為所有支援的幣種
    timeframes : list
        時間框架列表，預設為 ["15m", "1h"]
    local_cache_dir : str
        本地快取目錄
        
    返回：
    -----------
    dict
        格式: {
            "BTCUSDT": {
                "15m": DataFrame,
                "1h": DataFrame
            },
            ...
        }
    """
    
    if symbols is None:
        symbols = list(SYMBOLS.keys())
    
    if timeframes is None:
        timeframes = ["15m", "1h"]
    
    all_data = {}
    
    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"處理 {symbol}")
        logger.info(f"{'='*60}")
        
        all_data[symbol] = {}
        
        for timeframe in timeframes:
            if timeframe not in SYMBOLS[symbol]["timeframes"]:
                logger.warning(f"⚠️ {symbol} 不支援 {timeframe} 時間框架")
                continue
            
            df = download_kline_data(symbol, timeframe, local_cache_dir)
            if df is not None:
                all_data[symbol][timeframe] = df
            else:
                logger.warning(f"⚠️ 無法下載 {symbol} {timeframe} 資料")
    
    return all_data


def save_combined_data(
    data_dict: dict,
    output_dir: str = "downloaded_data"
) -> None:
    """
    將下載的資料保存為 CSV 檔案（便於後續處理）
    
    參數：
    -----------
    data_dict : dict
        由 download_multiple_symbols() 返回的資料字典
    output_dir : str
        輸出目錄
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for symbol, timeframe_data in data_dict.items():
        for timeframe, df in timeframe_data.items():
            output_file = f"{output_dir}/{symbol}_{timeframe}.csv"
            df.to_csv(output_file)
            logger.info(f"✅ 已保存: {output_file}")


if __name__ == "__main__":
    # 使用範例
    
    # 方案1：下載特定幣種和時間框架
    logger.info("開始下載加密貨幣資料\n")
    
    # 下載 BTC 和 ETH 的 15m 和 1h 資料
    selected_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    selected_timeframes = ["15m", "1h"]
    
    all_data = download_multiple_symbols(
        symbols=selected_symbols,
        timeframes=selected_timeframes,
        local_cache_dir="data_cache"
    )
    
    # 方案2：下載所有支援的幣種（需要更多時間和空間）
    # all_data = download_multiple_symbols(
    #     local_cache_dir="data_cache"
    # )
    
    # 保存資料
    logger.info(f"\n{'='*60}")
    logger.info("保存資料到本地")
    logger.info(f"{'='*60}\n")
    save_combined_data(all_data, output_dir="downloaded_data")
    
    # 顯示統計信息
    logger.info(f"\n{'='*60}")
    logger.info("下載統計")
    logger.info(f"{'='*60}")
    
    total_files = 0
    total_rows = 0
    
    for symbol, timeframe_data in all_data.items():
        for timeframe, df in timeframe_data.items():
            total_files += 1
            total_rows += len(df)
            logger.info(f"{symbol:12} {timeframe:4} - {len(df):8} 行")
    
    logger.info(f"{'='*60}")
    logger.info(f"總計: {total_files} 檔案, {total_rows} 行資料")
    logger.info(f"{'='*60}\n")
    
    # 快速查看資料示例
    if all_data:
        first_symbol = list(all_data.keys())[0]
        first_timeframe = list(all_data[first_symbol].keys())[0]
        sample_df = all_data[first_symbol][first_timeframe]
        
        logger.info(f"資料樣本 ({first_symbol} {first_timeframe}):\n")
        logger.info(sample_df.head(10))
