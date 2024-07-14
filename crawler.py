import requests
import sys
from pprint import pprint

def seize_content(url: str, onlyHead: bool = False) -> requests.Response:
    result = requests.get(url, )
    return result

def test(s: str):
    res = s.split(" ")
    pprint(res)



if __name__ == "__main__":
    test(sys.argv[1])
    