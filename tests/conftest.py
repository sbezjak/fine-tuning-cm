import datetime as dt
from pathlib import Path

from ft_cm.config import load_dotenv


def pytest_configure(config):
    """Rewrite the pytest-html output path to a unique UTC-timestamped file so no
    run overwrites another. A descriptive path passed explicitly is preserved;
    only the default is auto-renamed. Carried verbatim from P0-P5."""
    load_dotenv()
    if not hasattr(config.option, "htmlpath"):
        return
    config.option.self_contained_html = True
    if config.option.htmlpath not in (None, "reports/report.html"):
        return
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    config.option.htmlpath = str(reports / f"report-{ts}.html")
