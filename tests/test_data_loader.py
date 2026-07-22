"""单元测试 — src/data_loader.py 数据加载与预处理逻辑验证。

覆盖:
  - load_nav_data: CSV 加载、NaN ffill、全 NaN 行删除、截断至首个全有效日期
  - resample_weekly: 检测已有周频数据（中位间隔 6-8 天）、日频重采样为周频
  - classify_etfs: 进攻/防御 ETF 正确分类
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.data_loader import load_nav_data, resample_weekly, classify_etfs


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_csv(tmp_path):
    """创建一个临时 CSV 文件写入器，返回 (path, writer_func)。"""
    def _write(df: pd.DataFrame, path_name: str = "nav.csv") -> Path:
        p = tmp_path / path_name
        df.to_csv(p)
        return p
    return _write


@pytest.fixture
def clean_daily_nav():
    """构造干净的 5-ETF 日频净值序列（60 个交易日）。"""
    np.random.seed(123)
    dates = pd.bdate_range('2023-01-02', periods=60)
    cols = ['纳指ETF', '红利低波ETF', '中证500ETF', '黄金ETF', '国债ETF']
    rets = np.random.randn(60, 5) * 0.01
    prices = np.cumprod(1 + rets, axis=0) * 100
    return pd.DataFrame(prices, index=dates, columns=cols)


# ── load_nav_data 测试 ───────────────────────────────────────────────────────

class TestLoadNavData:
    def test_loads_csv_basic(self, tmp_csv, clean_daily_nav):
        """基本加载: CSV 读取后返回 DataFrame，列数正确。"""
        path = tmp_csv(clean_daily_nav)
        result = load_nav_data(path)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 5
        assert len(result) > 0

    def test_column_names(self, tmp_csv, clean_daily_nav):
        """加载后列名应为 ETFS 列表。"""
        path = tmp_csv(clean_daily_nav)
        result = load_nav_data(path)
        expected_cols = ['纳指ETF', '红利低波ETF', '中证500ETF', '黄金ETF', '国债ETF']
        assert list(result.columns) == expected_cols

    def test_ffill_single_etf_nan(self, tmp_csv, clean_daily_nav):
        """单只 ETF 缺失值应被 ffill 填充。"""
        df = clean_daily_nav.copy()
        # 在第 10~15 行将纳指ETF 设为 NaN
        df.iloc[10:16, 0] = np.nan
        path = tmp_csv(df)
        result = load_nav_data(path)
        # ffill 后不应有 NaN（截断后区域内）
        assert result['纳指ETF'].notna().all(), "ffill 未能填充单只 ETF 的 NaN"

    def test_drops_all_nan_rows(self, tmp_csv, clean_daily_nav):
        """全市场休市行（所有 ETF 为 NaN）应被删除。"""
        df = clean_daily_nav.copy()
        # 在第 5~8 行全部设为 NaN（模拟全市场休市）
        df.iloc[5:9, :] = np.nan
        path = tmp_csv(df)
        result = load_nav_data(path)
        # 不应有全 NaN 行
        assert not result.isna().all(axis=1).any(), "未删除全 NaN 行"

    def test_truncates_to_first_all_valid_date(self, tmp_csv):
        """截断至所有 ETF 都有数据之后的第一个日期。"""
        dates = pd.bdate_range('2023-01-02', periods=20)
        cols = ['纳指ETF', '红利低波ETF', '中证500ETF', '黄金ETF', '国债ETF']
        data = np.ones((20, 5)) * 100.0
        # 前 5 行国债ETF 为 NaN → first all-valid = index 5
        data[:5, 4] = np.nan
        df = pd.DataFrame(data, index=dates, columns=cols)
        path = tmp_csv(df)
        result = load_nav_data(path)
        # 结果的第一行日期应 >= 第 6 行日期（index=5）
        assert result.index[0] >= dates[5], (
            f"截断不正确: 首行={result.index[0]}, 期望>={dates[5]}"
        )

    def test_no_nan_after_processing(self, tmp_csv, clean_daily_nav):
        """经过完整处理后（截断区域内）不应有 NaN。"""
        df = clean_daily_nav.copy()
        # 散布一些 NaN
        df.iloc[3, 1] = np.nan
        df.iloc[7, 2] = np.nan
        path = tmp_csv(df)
        result = load_nav_data(path)
        assert result.notna().all().all(), "处理后仍有 NaN"

    def test_index_is_datetime(self, tmp_csv, clean_daily_nav):
        """索引应为 DatetimeIndex。"""
        path = tmp_csv(clean_daily_nav)
        result = load_nav_data(path)
        assert isinstance(result.index, pd.DatetimeIndex)


# ── resample_weekly 测试 ─────────────────────────────────────────────────────

class TestResampleWeekly:
    def test_detects_already_weekly(self):
        """数据已是周频（间隔 ~7 天）时，应直接返回副本。"""
        dates = pd.date_range('2023-01-02', periods=30, freq='W-MON')
        cols = ['纳指ETF', '中证500ETF']
        data = np.random.rand(30, 2) * 100
        df = pd.DataFrame(data, index=dates, columns=cols)
        result = resample_weekly(df)
        # 应返回相同数据
        assert len(result) == len(df), "已周频数据不应被重采样"
        pd.testing.assert_frame_equal(result, df)

    def test_weekly_median_gap_detection(self):
        """中位间隔在 6-8 天范围内时应被识别为周频。"""
        # 构造间隔恰好为 7 天的数据
        dates = pd.date_range('2023-01-02', periods=20, freq='7D')
        cols = ['A', 'B']
        data = np.ones((20, 2))
        df = pd.DataFrame(data, index=dates, columns=cols)
        result = resample_weekly(df)
        assert len(result) == 20

    def test_resamples_daily_to_weekly(self):
        """日频数据应被正确重采样为周频。"""
        dates = pd.bdate_range('2023-01-02', periods=100)  # ~100 交易日
        cols = ['纳指ETF', '国债ETF']
        np.random.seed(42)
        data = np.cumprod(1 + np.random.randn(100, 2) * 0.01, axis=0) * 100
        df = pd.DataFrame(data, index=dates, columns=cols)
        result = resample_weekly(df, anchor='W-MON')
        # 周频数据行数应明显少于日频
        assert len(result) < len(df), "重采样后行数应减少"
        # 大约 100 交易日 ≈ 20 周
        assert 15 <= len(result) <= 25, f"周频行数异常: {len(result)}"

    def test_resampled_has_no_all_nan_rows(self):
        """重采样后不应有全 NaN 行。"""
        dates = pd.bdate_range('2023-01-02', periods=50)
        cols = ['A', 'B']
        data = np.ones((50, 2)) * 100
        df = pd.DataFrame(data, index=dates, columns=cols)
        result = resample_weekly(df, anchor='W-MON')
        assert not result.isna().all(axis=1).any()

    def test_short_series_passthrough(self):
        """少于 3 行的序列应直接重采样（无法计算中位间隔）。"""
        dates = pd.bdate_range('2023-01-02', periods=2)
        cols = ['A']
        df = pd.DataFrame([[100.0], [101.0]], index=dates, columns=cols)
        result = resample_weekly(df)
        assert len(result) >= 1


# ── classify_etfs 测试 ───────────────────────────────────────────────────────

class TestClassifyEtfs:
    def test_default_5etf_classification(self):
        """默认 5-ETF 列表的分类应正确。"""
        names = ['纳指ETF', '红利低波ETF', '中证500ETF', '黄金ETF', '国债ETF']
        off_idx, def_idx, nasdaq_idx = classify_etfs(names)
        # 进攻层: 纳指(0), 中证500(2), 黄金(3)
        assert off_idx == [0, 2, 3], f"进攻层索引错误: {off_idx}"
        # 防御层: 红利低波(1), 国债(4)
        assert def_idx == [1, 4], f"防御层索引错误: {def_idx}"
        # 纳指代理: 纳指ETF(0)
        assert nasdaq_idx == 0, f"纳指代理索引错误: {nasdaq_idx}"

    def test_bond_etf_is_defensive(self):
        """债券类 ETF 应被分类为防御层。"""
        names = ['沪深300ETF', '信用债ETF', '可转债ETF']
        off_idx, def_idx, nasdaq_idx = classify_etfs(names)
        assert 1 in def_idx  # 信用债ETF
        assert 2 in def_idx  # 可转债ETF
        assert 0 in off_idx  # 沪深300ETF

    def test_hongli_is_defensive(self):
        """红利类 ETF 应被分类为防御层。"""
        names = ['中证500ETF', '红利ETF', '黄金ETF']
        off_idx, def_idx, nasdaq_idx = classify_etfs(names)
        assert 1 in def_idx  # 红利ETF
        assert 0 in off_idx  # 中证500ETF
        assert 2 in off_idx  # 黄金ETF

    def test_nasdaq_proxy_detection(self):
        """纳指相关 ETF 应被检测为纳指代理。"""
        names = ['国债ETF', '标普500ETF', '黄金ETF']
        off_idx, def_idx, nasdaq_idx = classify_etfs(names)
        assert nasdaq_idx == 1  # 标普500ETF

    def test_nasdaq_fallback_to_first_offensive(self):
        """无纳指相关 ETF 时，纳指代理应回退为第一个进攻 ETF。"""
        names = ['沪深300ETF', '国债ETF', '黄金ETF']
        off_idx, def_idx, nasdaq_idx = classify_etfs(names)
        assert nasdaq_idx == off_idx[0]  # 回退到第一个进攻 ETF

    def test_all_offensive(self):
        """全部进攻型 ETF 列表。"""
        names = ['纳指ETF', '中证500ETF', '黄金ETF', '创业板ETF']
        off_idx, def_idx, nasdaq_idx = classify_etfs(names)
        assert off_idx == [0, 1, 2, 3]
        assert def_idx == []

    def test_all_defensive(self):
        """全部防御型 ETF 列表。"""
        names = ['国债ETF', '红利低波ETF', '信用债ETF']
        off_idx, def_idx, nasdaq_idx = classify_etfs(names)
        assert off_idx == []
        assert def_idx == [0, 1, 2]
        # nasdaq_idx 回退为 0（无进攻 ETF 时）
        assert nasdaq_idx == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
