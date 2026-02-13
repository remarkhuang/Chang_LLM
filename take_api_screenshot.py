from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    
    # 5. Backend API Documentation
    page.goto("http://localhost:8000/docs")
    page.wait_for_load_state("networkidle")
    time.sleep(2)
    page.screenshot(path="D:/testcode/free_LLM/pagesview/5_api_docs.png", full_page=True)
    print("Screenshot 5: API documentation saved")
    
    browser.close()
    print("All screenshots completed!")
