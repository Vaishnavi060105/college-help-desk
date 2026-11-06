from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 1200})
    page.goto("http://localhost:8501", wait_until="networkidle")
    page.wait_for_timeout(2500)
    
    # Click sample query chip
    admission_btn = page.get_by_role("button", name="🎓 Admission")
    if admission_btn.count() > 0:
        admission_btn.click()
        page.wait_for_timeout(3500)
        
    page.screenshot(path="streamlit_chat_screenshot.png")
    print("Screenshot saved to streamlit_chat_screenshot.png")
    browser.close()
