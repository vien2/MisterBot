import requests
from bs4 import BeautifulSoup

def inspect():
    url = "https://gol.gg/game/stats/73783/page-game/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    print(f"Fetching {url}...")
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    
    # Check Red Header (KC should be Red if Vitality was Blue 2905)
    # Vitality (2905) was Blue in user's log.
    
    red_header = soup.find(class_="red-line-header")
    if not red_header:
        print("Red header not found!")
    else:
        print(f"Red Header Text: {red_header.get_text(strip=True)}")
        container = red_header.find_parent("div", class_="col-sm-6")
        if container:
            boxes = container.find_all(class_="score-box red_line")
            print(f"Found {len(boxes)} red score boxes:")
            for i, b in enumerate(boxes):
                print(f"  Box[{i}]: '{b.get_text(strip=True)}'")
        else:
            print("Red container not found")

    blue_header = soup.find(class_="blue-line-header")
    if blue_header:
        print(f"Blue Header Text: {blue_header.get_text(strip=True)}")
        container = blue_header.find_parent("div", class_="col-sm-6")
        if container:
            boxes = container.find_all(class_="score-box blue_line")
            print(f"Found {len(boxes)} blue score boxes:")
            for i, b in enumerate(boxes):
                print(f"  Box[{i}]: '{b.get_text(strip=True)}'")

if __name__ == "__main__":
    inspect()
