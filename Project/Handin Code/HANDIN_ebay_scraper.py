
"""
Author: Julian Valdman
Edited: 08-06-2022

File Purpose:
Collecting auction info from eBay auction page

"""

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys

import pandas as pd
import datetime
import time


url = "https://www.ebay.com/sch/i.html?_from=R40&_nkw=iphone+12+pro+max+256+gb&_in_kw=1&_ex_kw=&_sacat=0&LH_Complete=1&_udlo=&_udhi=&LH_Auction=1&_ftrt=901&_ftrv=1&_sabdlo=&_sabdhi=&_samilow=&_samihi=&_sadis=15&_stpos=&_sargn=-1%26saslc%3D1&_salic=1&_sop=13&_dmd=1&_ipg=240&_fosrp=1"

USERNAME = "julval281"
PASSWORD = "Valdman11"

#### Functions


def logon(driver):

    """ Logon to Ebay site """

    time.sleep(2)
    driver.find_element_by_id("userid").send_keys(USERNAME)
    driver.find_element_by_id("signin-continue-btn").click()

    time.sleep(2)
    driver.find_element_by_id("pass").send_keys(PASSWORD)
    driver.find_element_by_id("sgnBt").click()
    time.sleep(2)


def get_auctions_data_selenium(driver):

    """
    Traverse subpages to collect auction data set

    Returns
        A dictionary of auctions each with list of auction features
        A pandas dataframe containing bids for each auction
    """

    auctions_data = {}
    bidding_data = pd.DataFrame()

    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.CLASS_NAME, "sresult")))

    time.sleep(5)

    auctions = driver.find_elements_by_class_name("sresult")

    for idx, auction in enumerate(auctions[:]):

        try:

            # Clean up before each iteration
            while len(driver.window_handles) > 1:
                driver.switch_to.window(driver.window_handles[-1])
                driver.close()
                driver.switch_to.window(driver.window_handles[-1])
            driver.switch_to.window(driver.window_handles[-1])

            bids = auction.find_elements_by_class_name("lvformat")[0].text.strip()
            bids = int(bids.split(" ")[0])

            title = auction.find_elements_by_class_name("vip")[0].text.strip()

            # Auction needs to have at least 3 bids
            if bids >= 3:

                title_element = auction.find_elements_by_class_name("vip")[0]
                actions = ActionChains(driver)

                # Open 2 new tabs with same page for now
                actions.key_down(Keys.CONTROL).click(title_element).key_up(Keys.CONTROL).perform()
                driver.switch_to.window(driver.window_handles[-1])

                driver.switch_to.window(driver.window_handles[0])
                actions.key_down(Keys.CONTROL).click(title_element).key_up(Keys.CONTROL).perform()

                # If opened more than 2 new ones, close the excessive
                while len(driver.window_handles) > 3:
                    driver.switch_to.window(driver.window_handles[-1])
                    driver.close()
                    driver.switch_to.window(driver.window_handles[-1])

                #### First auction page
                try:
                    condition = driver.find_element_by_id("vi-itm-cond").text.strip()
                except Exception as e:
                    print(f"ERROR: ({idx}) - Error on first page. Continuing ... ")
                    continue

                try:
                    conditionDesc = driver.find_elements_by_class_name("topItmCndDscMsg")[0].text.strip()
                except Exception:
                    conditionDesc = "N/A"

                ended_time = driver.find_element_by_id("bb_tlft").text.strip()
                ended_time = datetime.datetime.strptime(ended_time, '%b %d, %Y , %I:%M%p')
                ended_time_weekday = ended_time.weekday()
                ended_time_weekend = 1 if ended_time_weekday in [5, 6] else 0

                price_original = driver.find_elements_by_class_name("notranslate")[0].text
                price_dkk = driver.find_element_by_id("convbidPrice").text.strip()
                price_dkk = int(price_dkk.replace("DKK ", "").split(".")[0].replace(",", ""))
                seller_location = driver.find_elements_by_xpath("//div[text()='Located in:']/following-sibling::div")[0].text

                # Get Feedback Rating from Seller
                driver.switch_to.window(driver.window_handles[1])
                driver.find_elements_by_class_name("mbg-nw")[0].click()
                time.sleep(2)

                try:
                    pct_feedback = driver.find_elements_by_class_name("perctg")[0].text
                    pct_feedback = float(pct_feedback.replace("% positive feedback", ""))
                except Exception:
                    print(f"INFO: ({idx}) - Could not find seller pct rating.")
                    pct_feedback = "N/A"

                seller_country = driver.find_elements_by_class_name("mem_loc")[0].text
                try:
                    num_pos = driver.find_elements_by_xpath("//a[@title='Positive']")[0].find_elements_by_class_name("num")[0].text
                    num_pos = int(num_pos.replace(",", ""))

                    num_neutral = driver.find_elements_by_xpath("//a[@title='Neutral']")[0].find_elements_by_class_name("num")[0].text
                    num_neutral = int(num_neutral.replace(",", ""))

                    num_neg = driver.find_elements_by_xpath("//a[@title='Negative']")[0].find_elements_by_class_name("num")[0].text
                    num_neg = int(num_neg.replace(",", ""))

                except Exception:
                    print(f"INFO: ({idx}) - Could not find seller rating.")
                    num_pos = "N/A"
                    num_neutral = "N/A"
                    num_neg = "N/A"

                seller_member_since = driver.find_elements_by_id("member_info")[0].find_elements_by_xpath("//span[text()='Member since: ']/following-sibling::span")[0].text
                seller_member_since = datetime.datetime.strptime(seller_member_since, '%b %d, %Y')
                seller_member_duration = (datetime.datetime.today() - seller_member_since).days

                # Get bidder history and more info
                driver.switch_to.window(driver.window_handles[2])

                try:
                    time.sleep(1)
                    driver.find_elements_by_id("vi-VR-bid-lnk-")[0].click()
                except Exception:
                    print(f"INFO: ({idx}) - Could not navigate to bidder history page.")
                    continue

                # Check whether login is necessary
                header_not_found = driver.find_elements_by_class_name("app-main-container-upgrade__item_card") == []
                if header_not_found:
                    logon(driver)

                time.sleep(2)

                num_bidders = driver.find_elements_by_class_name("app-bid-info-upgrade_wrapper")[0].find_elements_by_tag_name("li")[1].find_elements_by_tag_name("span")[3].text
                duration = driver.find_elements_by_class_name("app-bid-info-upgrade_wrapper")[0].find_elements_by_tag_name("li")[4].find_elements_by_tag_name("span")[3].text
                duration = int(duration.split(" ")[0])

                bidder_history = pd.read_html(driver.find_elements_by_xpath("//div[@class='app-container-view-upgrade__content_table']")[0].get_attribute('innerHTML'))[0]
                bidder_history["link"] = driver.current_url
                bidding_data = bidder_history if bidding_data.shape[1] < 2 else pd.concat([bidding_data, bidder_history], ignore_index=False)

                # Close the (2) opened tabs
                driver.switch_to.window(driver.window_handles[-1])
                driver.close()
                driver.switch_to.window(driver.window_handles[-1])
                driver.close()

                # Append data
                auctions_data[idx] = {"title": title}
                auctions_data[idx]["condition"] = condition
                auctions_data[idx]["conditionDescription"] = conditionDesc
                auctions_data[idx]["ended_time"] = ended_time
                auctions_data[idx]["ended_time_weekday"] = ended_time_weekday
                auctions_data[idx]["ended_time_weekend"] = ended_time_weekend
                auctions_data[idx]["price"] = price_original
                auctions_data[idx]["price_dkk"] = price_dkk
                auctions_data[idx]["seller_location"] = seller_location
                auctions_data[idx]["seller_country"] = seller_country
                auctions_data[idx]["seller_feedback_pct"] = pct_feedback
                auctions_data[idx]["seller_feedback_positive"] = num_pos
                auctions_data[idx]["seller_feedback_neutral"] = num_neutral
                auctions_data[idx]["seller_feedback_negative"] = num_neg
                auctions_data[idx]["seller_member_since"] = seller_member_since
                auctions_data[idx]["seller_member_duration"] = seller_member_duration
                auctions_data[idx]["num_bidders"] = num_bidders
                auctions_data[idx]["bids"] = bids
                auctions_data[idx]["duration"] = duration

                print(f"INFO: ({idx}) - Success")

        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            break

        except Exception as e:
            print(f"{idx} - {str(e)} - Error")

    return auctions_data, bidding_data


if __name__ == "__main__":

    driver = webdriver.Chrome(ChromeDriverManager().install())

    driver.get(url)
    driver.maximize_window()
    auctions_data_dict, bidding_data_df = get_auctions_data_selenium(driver)

    driver.quit()

    auction_data_df = pd.DataFrame.from_dict(auctions_data_dict, orient="index")

    auction_data_df = auction_data_df.drop_duplicates()
    bidding_data = bidding_data_df.drop_duplicates()

    auction_data_df.to_excel("data/COPY_auctions_data.xlsx", index=False)
    bidding_data.to_excel("data/COPY_bidding_data.xlsx", index=False)
