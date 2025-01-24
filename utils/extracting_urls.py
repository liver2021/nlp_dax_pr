from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time

def News_Scrapper(filename="default"):
    Date = []
    Header = []
    Link = []
    tick = 0
    filename = filename

    driver = webdriver.Firefox()
    driver.get("https://www.boerse-frankfurt.de/nachrichten/aktien")
    wait = WebDriverWait(driver, 30)
    tbody = wait.until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
    rows = tbody.find_elements(By.TAG_NAME, "tr")

    for x in range(2, 905):
        start_time = time.time()
        tbody = wait.until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
        rows = tbody.find_elements(By.TAG_NAME, "tr")

        for count, row in enumerate(rows):
            try:
                links = row.find_elements(By.TAG_NAME, "a")

                for link in links:
                    link_url = link.get_attribute('href')
                    Link.append(link_url)
                    tick = tick + 1

                tds = row.find_elements(By.TAG_NAME, "td")

                for td_count, td in enumerate(tds):
                    if td_count == 0:
                        td_text = td.text
                        Date.append(td_text)

                    else:
                        td_text = td.text
                        Header.append(td_text)


            except Exception as e:
                print(f"An error occurred: {str(e)}")

        button_title = f"Zeige Seite {x}"
        button = wait.until(EC.visibility_of_element_located((By.XPATH, f"//button[@title='{button_title}']")))
        driver.execute_script("arguments[0].click();", button)
        time.sleep(2)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Page: {x - 1}/904 Type:Url Time:{elapsed_time}")

    def content_exists(file_content, entry):
        return entry in file_content

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except FileNotFoundError:
        file_content = ""

    with open(filename, 'a', encoding='utf-8') as file:
        for date, header, link in zip(Date, Header, Link):
            entry = f"{date}\t{link}\t{header}\n"

            if not content_exists(file_content, entry):
                file.write(entry)
                file_content += entry


output_file = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/urls_data.txt"

News_Scrapper(filename=output_file)
"""Extracts Headers for Daily News from www.boerse-frankfurt.de
 Args:
     filename= Path for the Outputfile which contains
 Example: Output: [01.02.2019, (Link to full article), DAX reached new all time high]
 
     """
