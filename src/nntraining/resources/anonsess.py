import requests


def get_session():
    session = requests.Session()
    # tor uses the 9050 port as the default socks port, TBB 9150
    session.proxies = {'http':  'socks5://127.0.0.1:9150',
                       'https': 'socks5://127.0.0.1:9150'}
    return session
