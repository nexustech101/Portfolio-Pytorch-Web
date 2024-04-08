import requests
import numpy as np

def get_trending_repos():
    url = 'https://github.com/trending'
    with requests.get(url, verify=False) as res:
        code = res.status_code
        if code == 200:
            html = res.content
            print(html)
            
if __name__ == '__main__':
    # get_trending_repos()
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.5, 0.1, 0.4]
    ])

    v = np.array([0.3, 0.4, 0.3])
    # Reshape b to ensure it's treated as a column vector
    v = v.reshape(-1, 1)
    c = P @ v
    print(c)
    