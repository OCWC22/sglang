import inspect
import unittest
from types import SimpleNamespace


def _lmcache_mp_enabled(server_args) -> bool:
    return bool(
        getattr(server_args, "lmcache_mp_host", None)
        or getattr(server_args, "lmcache_mp_port", None)
    )


def _build_connector_kwargs(connector_cls, kwargs):
    signature = inspect.signature(connector_cls)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _resolve_lmcache_mode_and_connector_name(connector_cls, mp_requested: bool):
    connector_name = connector_cls.__name__
    mode = (
        "mp"
        if mp_requested and connector_name == "LMCacheMPLayerwiseConnector"
        else "embedded"
    )
    return mode, connector_name


def _build_lmcache_metric_labels(
    server_args, cache_type: str, mode: str, connector_name: str
):
    return {
        "cache_type": cache_type,
        "enable_lmcache": str(bool(getattr(server_args, "enable_lmcache", False))).lower(),
        "lmcache_mode": mode,
        "lmcache_connector": connector_name,
        "lmcache_mp_host": str(getattr(server_args, "lmcache_mp_host", "") or ""),
        "lmcache_mp_port": str(getattr(server_args, "lmcache_mp_port", "") or ""),
    }


class _ExplicitConnector:
    def __init__(self, sgl_config=None, lmcache_mp_host=None):
        pass


class _KwargsConnector:
    def __init__(self, **kwargs):
        pass


class LMCacheMPLayerwiseConnector:
    pass


class LMCacheLayerwiseConnector:
    pass


class _Metrics:
    def __init__(self):
        self.releases = []
        self.aborts = []

    def increment_release_call(self, reason):
        self.releases.append(reason)

    def increment_abort_call(self, reason):
        self.aborts.append(reason)


class _ReleaseConnector:
    def __init__(self):
        self.calls = []

    def release_request(self, rid=None, reason=None):
        self.calls.append((rid, reason))


class _LifecycleHarness:
    def __init__(self):
        self.lmcache_connector = _ReleaseConnector()
        self.lmcache_metrics_collector = _Metrics()

    def release_lmcache_request(self, rid=None, reason="release"):
        connector = getattr(self, "lmcache_connector", None)
        if connector is None:
            return
        released = False
        for method_name in (
            "release_request",
            "abort_request",
            "end_request",
            "close_request",
            "cleanup_request",
        ):
            method = getattr(connector, method_name, None)
            if method is None:
                continue
            kwargs = {"rid": rid, "reason": reason}
            method_kwargs = _build_connector_kwargs(method, kwargs)
            method(**method_kwargs)
            released = True
            break
        if released and self.lmcache_metrics_collector is not None:
            self.lmcache_metrics_collector.increment_release_call(reason)
            if reason in ("abort", "preempt", "timeout"):
                self.lmcache_metrics_collector.increment_abort_call(reason)


class TestLMCacheMPHelpers(unittest.TestCase):
    def test_lmcache_mp_enabled_by_host_or_port(self):
        args = SimpleNamespace(lmcache_mp_host=None, lmcache_mp_port=None)
        self.assertFalse(_lmcache_mp_enabled(args))
        args.lmcache_mp_host = "127.0.0.1"
        self.assertTrue(_lmcache_mp_enabled(args))
        args.lmcache_mp_host = None
        args.lmcache_mp_port = 6555
        self.assertTrue(_lmcache_mp_enabled(args))

    def test_build_connector_kwargs_filters_unknown_args(self):
        kwargs = {
            "sgl_config": object(),
            "lmcache_mp_host": "127.0.0.1",
            "unknown": True,
        }
        self.assertEqual(
            set(_build_connector_kwargs(_ExplicitConnector, kwargs).keys()),
            {"sgl_config", "lmcache_mp_host"},
        )
        self.assertEqual(_build_connector_kwargs(_KwargsConnector, kwargs), kwargs)

    def test_mp_mode_requires_actual_mp_connector(self):
        self.assertEqual(
            _resolve_lmcache_mode_and_connector_name(LMCacheMPLayerwiseConnector, True),
            ("mp", "LMCacheMPLayerwiseConnector"),
        )
        self.assertEqual(
            _resolve_lmcache_mode_and_connector_name(LMCacheLayerwiseConnector, True),
            ("embedded", "LMCacheLayerwiseConnector"),
        )
        self.assertEqual(
            _resolve_lmcache_mode_and_connector_name(LMCacheMPLayerwiseConnector, False),
            ("embedded", "LMCacheMPLayerwiseConnector"),
        )

    def test_metric_labels_include_enable_lmcache_and_mp_endpoint(self):
        args = SimpleNamespace(
            enable_lmcache=True,
            lmcache_mp_host="127.0.0.1",
            lmcache_mp_port=6555,
        )
        labels = _build_lmcache_metric_labels(
            args, "LMCRadixCache", "mp", "LMCacheMPLayerwiseConnector"
        )
        self.assertEqual(labels["enable_lmcache"], "true")
        self.assertEqual(labels["lmcache_mp_host"], "127.0.0.1")
        self.assertEqual(labels["lmcache_mp_port"], "6555")
        self.assertEqual(labels["lmcache_mode"], "mp")
        self.assertEqual(labels["lmcache_connector"], "LMCacheMPLayerwiseConnector")

    def test_release_lifecycle_records_abort_reason(self):
        harness = _LifecycleHarness()
        harness.release_lmcache_request("rid-1", reason="abort")
        self.assertEqual(harness.lmcache_connector.calls, [("rid-1", "abort")])
        self.assertEqual(harness.lmcache_metrics_collector.releases, ["abort"])
        self.assertEqual(harness.lmcache_metrics_collector.aborts, ["abort"])


if __name__ == "__main__":
    unittest.main()
