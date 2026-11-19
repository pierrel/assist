"""Host agent URL fetching with the control-service fence at socket creation."""
import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.util.connection import create_connection

from assist.control_api import resolve_target


class _CheckedConnection:
    def _new_conn(self):
        address = resolve_target(self.host, self.port)
        return create_connection((address, self.port), self.timeout,
                                 source_address=self.source_address,
                                 socket_options=self.socket_options)


class _HttpConnection(_CheckedConnection, HTTPConnection):
    pass


class _HttpsConnection(_CheckedConnection, HTTPSConnection):
    pass


class _HttpPool(HTTPConnectionPool):
    ConnectionCls = _HttpConnection


class _HttpsPool(HTTPSConnectionPool):
    ConnectionCls = _HttpsConnection


class _Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        # A private mapping keeps other host HTTP clients unchanged.
        self.poolmanager.pool_classes_by_scheme = {
            "http": _HttpPool, "https": _HttpsPool}


def get(url: str, **kwargs):
    """GET through checked connections, including every redirect destination.

    Keep the response's session alive until its streaming body is closed.
    Environment proxies cannot replace the checked connection transport.
    """
    session = requests.Session()
    session.trust_env = False
    session.mount("http://", _Adapter())
    session.mount("https://", _Adapter())
    try:
        response = session.get(url, **kwargs)
    except BaseException:
        session.close()
        raise
    close = response.close

    def close_response():
        try:
            close()
        finally:
            session.close()

    response.close = close_response
    return response
