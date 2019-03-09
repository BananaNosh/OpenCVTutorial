from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from time import sleep
from urllib.request import urlretrieve
import os


def load_credentials(filename):
    with open(filename, "r") as f:
        mail, password = f.readlines()
    return mail, password


def setup_browser():
    opts = Options()
    opts.headless = True
    browser = Firefox(options=opts)
    browser.get("https://stallcatchers.com/virtualMicroscope")
    print(browser.current_url)
    if browser.current_url.endswith("login"):
        mail, password = load_credentials("./credentials")
        email_field = browser.find_element_by_id("email")
        email_field.send_keys(mail)
        password_field = browser.find_element_by_id("password")
        password_field.send_keys(password)
        btn = browser.find_element_by_class_name("btn-primary")
        btn.click()
    try:
        skip_btn = browser.find_element_by_class_name("introjs-skipbutton")
        skip_btn.click()
        sleep(1)
    except NoSuchElementException:
        pass
    return browser


def evaluate_and_store(browser):
    flow_btn = browser.find_element_by_id("flowing")
    answer_space = browser.find_element_by_id("answerSpace")
    wait_for_click(flow_btn)
    try_again_next_btn = browser.find_element_by_id("nextMovie")
    video = browser.find_element_by_id("movieDiv").find_element_by_class_name("embed-responsive-item")
    video_link = video.get_attribute("src")
    counter = 0
    while answer_space.text == "":
        sleep(0.1)
        counter += 1
        if counter == 100:
            return evaluate_and_store(browser)
    if answer_space.text.startswith("Incorrect"):
        print("stalled")
        stalled = True
    elif answer_space.text.startswith("Correct"):
        print("flowing")
        stalled = False
    else:
        raise ValueError("Strange text in Answer space")
    video_folder = os.path.join("data", "learning", "stalled" if stalled else "flowing")
    _, video_name = os.path.split(video_link)
    print(f"Start downloading {video_name}")
    urlretrieve(video_link, os.path.join(video_folder, video_name))
    print("Finished")
    if stalled:
        try:
            browser.find_element_by_class_name("redSliderPointers")
        except NoSuchElementException:
            print("No stalls marked so far")
            return False

        stall_positions = []
        actions = ActionChains(browser)
        actions.send_keys(Keys.ARROW_RIGHT)
        frame_field = browser.find_element_by_id("frame")
        frame_value = int(frame_field.get_attribute("value"))
        while True:
            try:
                circle = browser.find_element_by_class_name("smallCircle")
                stall_positions.append((frame_value, circle.location))
            except NoSuchElementException:
                pass
            actions.perform()
            new_value = int(frame_field.get_attribute("value"))
            if new_value < frame_value:
                break
            frame_value = new_value
        print(stall_positions)
        info_filename = os.path.join(video_folder, os.path.splitext(video_name)[0] + "_position_info.txt")
        with open(info_filename, "w+") as f:
            for time, position in stall_positions:
                f.write(f"{time} {position['x']} {position['y']}\n")
    should_try_again = try_again_next_btn.get_attribute("value") == "Try again"
    wait_for_click(try_again_next_btn)
    if should_try_again:
        stalled_btn = browser.find_element_by_id("stalled")
        wait_for_click(stalled_btn)
        video.click()
        wait_for_click(try_again_next_btn)
    return True


def wait_for_click(flow_btn):
    while True:
        try:
            flow_btn.click()
            break
        except ElementNotInteractableException:
            pass


if __name__ == '__main__':
    browser = setup_browser()
    for i in range(100):
        print("step", i)
        if not evaluate_and_store(browser):
            exit(1)
        sleep(1)
    browser.close()
