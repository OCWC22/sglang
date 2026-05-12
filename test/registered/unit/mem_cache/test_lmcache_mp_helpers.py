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


class _ExplicitConnector:
    def __init__(self, sgl_config=None, lmcache_mp_host=None):
        pass


class _KwargsConnector:
    def __init__(self, **kwargs):
        pass


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


if __name__ == "__main__":
    unittest.main()
