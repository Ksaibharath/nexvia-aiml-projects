"""Redis cache with model_version key isolation, circuit-breaker backoff, and resilient local TTL fallback."""
import json
import os
import pathlib
import time
import redis


class Cache:
    def __init__(self, ttl: int = 3600):
        self.ttl = int(ttl)
        self.r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            socket_connect_timeout=0.1,
            socket_timeout=0.1,
            decode_responses=True,
        )
        self._local_store: dict[str, tuple[float, str]] = {}
        self._redis_down_until: float = 0.0
        self._cached_ver: str | None = None
        self._ver_check_ts: float = 0.0
        self.hits = 0
        self.misses = 0
        self._version_file = pathlib.Path("artifacts/model_version.txt")

    def _redis_ok(self) -> bool:
        return time.monotonic() >= self._redis_down_until

    def _mark_down(self) -> None:
        self._redis_down_until = time.monotonic() + 15.0

    def version(self) -> str:
        now = time.monotonic()
        if self._cached_ver is not None and (now - self._ver_check_ts) < 5.0:
            return self._cached_ver
        self._ver_check_ts = now
        if self._redis_ok():
            try:
                v = self.r.get("model_version")
                if v:
                    self._cached_ver = str(v)
                    return self._cached_ver
            except redis.RedisError:
                self._mark_down()
        if self._version_file.exists():
            self._cached_ver = self._version_file.read_text(encoding="utf-8").strip() or "1"
            return self._cached_ver
        self._cached_ver = str(os.getenv("MODEL_VERSION", "1"))
        return self._cached_ver

    def set_version(self, ver: int | str) -> None:
        ver_s = str(ver)
        self._cached_ver = ver_s
        self._ver_check_ts = time.monotonic()
        if self._redis_ok():
            try:
                self.r.set("model_version", ver_s)
            except redis.RedisError:
                self._mark_down()
        self._version_file.parent.mkdir(parents=True, exist_ok=True)
        self._version_file.write_text(ver_s, encoding="utf-8")

    def key(self, user_id: int, k: int, variant: str, extra: str = "") -> str:
        base = f"rec:v{self.version()}:{variant}:{user_id}:{k}"
        return f"{base}:{extra}" if extra else base

    def get(self, user_id: int, k: int, variant: str, extra: str = ""):
        k_str = self.key(user_id, k, variant, extra)
        if self._redis_ok():
            try:
                v = self.r.get(k_str)
                if v is not None:
                    self.hits += 1
                    return json.loads(v)
                self.misses += 1
                return None
            except redis.RedisError:
                self._mark_down()

        entry = self._local_store.get(k_str)
        if entry is not None:
            exp_ts, raw = entry
            if time.time() < exp_ts:
                self.hits += 1
                return json.loads(raw)
            self._local_store.pop(k_str, None)
        self.misses += 1
        return None

    def set(self, user_id: int, k: int, variant: str, items, extra: str = "") -> None:
        k_str = self.key(user_id, k, variant, extra)
        payload = json.dumps(items)
        if self._redis_ok():
            try:
                self.r.setex(k_str, self.ttl, payload)
                return
            except redis.RedisError:
                self._mark_down()

        if len(self._local_store) > 10_000:
            self._local_store.clear()
        self._local_store[k_str] = (time.time() + self.ttl, payload)

    def clear_local(self) -> None:
        self._local_store.clear()

    def stats(self) -> dict:
        backend = "local_memory_fallback"
        redis_hits, redis_misses = None, None
        if self._redis_ok():
            try:
                info = self.r.info("stats")
                backend = "redis"
                redis_hits = info.get("keyspace_hits", 0)
                redis_misses = info.get("keyspace_misses", 0)
            except redis.RedisError:
                self._mark_down()
        total = self.hits + self.misses
        return {
            "backend": backend,
            "model_version": self.version(),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
            "redis_keyspace_hits": redis_hits,
            "redis_keyspace_misses": redis_misses,
        }
