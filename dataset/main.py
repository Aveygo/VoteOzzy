import requests, logging, time, json
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.root.setLevel(logging.INFO)

BASE_URL = "https://theyvoteforyou.org.au"

pol_soup = BeautifulSoup(requests.get(BASE_URL+ "/people").text , features="lxml")

result = {"politicians": {}, "policies": {}}

for politician_link in pol_soup.find_all('a'):
    

    politician_url = politician_link.get('href')
    
    if (not "/people/" in politician_url) or (not politician_url.count('/') == 4):
        continue

    logger.info(politician_url)

    politician_name = politician_link.find("h2").text
    result["politicians"][politician_name] = {
        "party": politician_link.find("p", {"class": "member-role"}).text.strip().replace("\n", " ").replace("  ", " "),
        "stances": {}
    }

    time.sleep(1)
    logger.info(f"Going to {BASE_URL + politician_url}")
    politician = BeautifulSoup(requests.get(BASE_URL + politician_url).text, features="lxml")
    
    #print(len(politician.find_all("div", class_="policy-comparision-block")))

    for position in politician.find_all("div", {"class":"policy-comparision-block"}):
        #print(position.text)
        position_text = position.find("h3", {"class": "policy-comparision-position"}).text
        logging.info(position_text)

        result["politicians"][politician_name]["stances"][position_text] = []

        logger.info(f"Analyzing {politician_name}'s '{position_text}' stances...")
        
        for policy in position.find_all('a'):
            policy_url = policy.get("href")

            if not "/policies/" in policy_url:
                continue

            policy_name = policy.text.strip()

            result["politicians"][politician_name]["stances"][position_text].append(policy_name)

            with open("result.json", "w") as f:
                json.dump(result, f, indent="\t")

            if not policy_name in result["policies"]:

                if not (policy_url.startswith(politician_url) and "/policies/" in policy_url):
                    continue
                
                time.sleep(1)
                policy_soup = BeautifulSoup(requests.get(BASE_URL + '/policies/' + policy_url.split("/")[-1]).text, features="lxml")

                description = policy_soup.find("div", {"class": "policytext"}).text.strip()
                result["policies"][policy_name] = description.replace("\n", " ")

                logger.info(f"Added the '{policy_name}' policy to the result")

                with open("result.json", "w") as f:
                    json.dump(result, f, indent="\t")

with open("result.json", "w") as f:
    json.dump(result, f, indent="\t")