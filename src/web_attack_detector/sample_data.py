"""Small labelled sample set used for smoke tests and the zero-data demo path."""

from __future__ import annotations

import pandas as pd

from web_attack_detector.config import AttackClass

_BENIGN: tuple[str, ...] = (
    "GET /index.php?id=1",
    "GET /search?q=python programming",
    "POST /login HTTP/1.1\nContent-Type: application/x-www-form-urlencoded\n\n"
    "username=user&password=pass",
    "GET /api/users/profile",
    "GET /static/css/style.css",
    "POST /contact HTTP/1.1\nContent-Type: application/json\n\n"
    '{"name":"John","email":"john@example.com"}',
)

_XSS: tuple[str, ...] = (
    '<script>alert("XSS")</script>',
    '<img src=x onerror=alert("XSS")>',
    '<iframe src="javascript:alert(1)"></iframe>',
    "javascript:alert(document.cookie)",
    '<svg onload=alert("XSS")>',
    '<input type="text" value="" onfocus="alert(\'XSS\')" autofocus>',
)

_CSRF: tuple[str, ...] = (
    "POST /transfer HTTP/1.1\nContent-Type: application/x-www-form-urlencoded\n\n"
    "amount=1000&to=attacker",
    "GET /admin/delete_user?id=123",
    "POST /settings/change_password HTTP/1.1\n\nnew_password=hacked&confirm=hacked",
    "DELETE /api/user/456 HTTP/1.1",
    "PUT /admin/users/789 HTTP/1.1\n\nrole=admin",
    "POST /payment/process HTTP/1.1\n\namount=999&account=evil",
)

_SQLI: tuple[str, ...] = (
    "' OR '1'='1' --",
    "admin' AND (SELECT COUNT(*) FROM users) > 0 --",
    "1' UNION SELECT username,password FROM users--",
    "'; DROP TABLE users; --",
    "1' OR SLEEP(5)--",
    "' OR 1=1 LIMIT 1 OFFSET 0 --",
)


def create_sample_data() -> pd.DataFrame:
    """Return a 24-row, 4-class frame with ``payload``, ``label`` and ``attack_type``.

    This exists so ``train.py`` runs with no dataset present. It is far too small
    to train a useful model and is duplicated internally by
    :func:`replicate_for_smoke_test`.
    """
    payloads = [*_BENIGN, *_XSS, *_CSRF, *_SQLI]
    labels = [
        *(int(AttackClass.BENIGN) for _ in _BENIGN),
        *(int(AttackClass.XSS) for _ in _XSS),
        *(int(AttackClass.CSRF) for _ in _CSRF),
        *(int(AttackClass.SQLI) for _ in _SQLI),
    ]
    return pd.DataFrame(
        {
            "payload": payloads,
            "label": labels,
            "attack_type": [
                name.lower()
                for name, count in zip(
                    ("benign", "xss", "csrf", "sqli"),
                    (len(_BENIGN), len(_XSS), len(_CSRF), len(_SQLI)),
                    strict=True,
                )
                for _ in range(count)
            ],
        },
    )


def replicate_for_smoke_test(df: pd.DataFrame, repeats: int = 50) -> pd.DataFrame:
    """Concatenate ``df`` with itself ``repeats`` times for smoke tests.

    Args:
        df: Frame to replicate.
        repeats: Total number of copies.

    Returns:
        The replicated frame with a reset index.
    """
    return pd.concat([df] * repeats, ignore_index=True)
