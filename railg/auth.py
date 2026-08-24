"""轻量鉴权。

个人项目默认 `auth.enabled=false`,所有请求以匿名 public 身份通过 ——
但 Identity/Principal 模型始终在链路里,所以打开开关就能生效,
不需要改检索或索引代码 —— 权限模型与是否启用是解耦的。

JWT 用标准库实现 HS256,不引入额外依赖(个人项目够用;
要接 Keycloak/OIDC 时,把 `identity_from_token` 换成校验 IdP 签发的
token 即可,下游 principals() 的用法不变)。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time

from railg.config import Settings, get_settings
from railg.schema.document import ANONYMOUS, Identity

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 最小 JWT (HS256)
# --------------------------------------------------------------------------- #
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def encode_token(payload: dict, secret: str) -> str:
    header = _b64e(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}".encode()
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64e(sig)}"


def decode_token(token: str, secret: str) -> dict | None:
    try:
        header, body, sig = token.split(".")
    except ValueError:
        return None
    expected = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64d(sig), expected):
        return None
    try:
        payload = json.loads(_b64d(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload


# --------------------------------------------------------------------------- #
# 口令
# --------------------------------------------------------------------------- #
def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()


def verify_password(password: str, expected: str, salt: str) -> bool:
    return hmac.compare_digest(hash_password(password, salt), expected)


# --------------------------------------------------------------------------- #
# 身份
# --------------------------------------------------------------------------- #
class AuthManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.auth
        self.enabled = cfg.enabled
        self._secret = cfg.jwt_secret
        if self.enabled and not self._secret:
            # 没配就临时生成:重启后旧 token 失效,对个人项目可接受,
            # 但会打日志提醒,避免用户以为是 bug。
            self._secret = secrets.token_urlsafe(32)
            logger.warning(
                "未设置 RAILG_JWT_SECRET,已生成临时密钥 —— 重启后需要重新登录"
            )
        if self.enabled and not cfg.admin_password:
            raise RuntimeError(
                "auth.enabled=true 但未设置 RAILG_ADMIN_PASSWORD"
            )

    def login(self, username: str, password: str) -> str | None:
        cfg = self.settings.auth
        if not self.enabled:
            return None
        ok = hmac.compare_digest(username, cfg.admin_user) and hmac.compare_digest(
            password, cfg.admin_password
        )
        if not ok:
            return None
        now = int(time.time())
        return encode_token(
            {
                "sub": username,
                "name": username,
                "roles": ["admin"],
                "groups": [],
                "iat": now,
                "exp": now + cfg.jwt_expire_minutes * 60,
            },
            self._secret,
        )

    def identity_from_token(self, token: str | None) -> Identity:
        """没开鉴权 → 匿名;开了但 token 无效 → 匿名(只能看 public 文档)。"""
        if not self.enabled:
            return ANONYMOUS
        if not token:
            return ANONYMOUS
        payload = decode_token(token.removeprefix("Bearer ").strip(), self._secret)
        if not payload:
            return ANONYMOUS
        return Identity(
            sub=payload.get("sub", "anonymous"),
            display_name=payload.get("name", payload.get("sub", "anonymous")),
            groups=list(payload.get("groups", [])),
            roles=list(payload.get("roles", [])),
        )


_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    global _manager
    if _manager is None:
        _manager = AuthManager()
    return _manager
