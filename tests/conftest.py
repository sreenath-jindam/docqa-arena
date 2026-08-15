import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src.types import Document  # noqa: E402


@pytest.fixture
def doc():
    content = (
        "# Refunds\n\n"
        "A refund returns money to a customer. Refunds cannot exceed the charge.\n\n"
        "## Timing\n\n"
        "Card refunds take five to ten business days. The issuing bank controls this.\n\n"
        "## Fees\n\n"
        "The processing fee is not returned when you refund a charge.\n"
    )
    return Document(id="refunds.md", source="refunds.md", content=content)


@pytest.fixture
def long_doc():
    paragraph = "Sentence one here. Sentence two follows it. Sentence three closes the paragraph.\n\n"
    return Document(id="long.md", source="long.md", content=paragraph * 40)
