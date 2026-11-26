import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, 'jp')):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

import hd  # noqa: E402

_ME_US = list(hd.ME)      # 日本版の推定（estimate_jp.build）は hd.ME を書き換えるので、テストごとに米国の値に戻す


@pytest.fixture(autouse=True)
def _restore_hd_me():
    hd.ME = list(_ME_US)
    yield
    hd.ME = list(_ME_US)
