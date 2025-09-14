from playwright.sync_api import sync_playwright
import time

def test_rag_ui():
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Go to the main page
        page.goto("http://127.0.0.1:8000")
        page.wait_for_load_state("networkidle")

        # Take initial screenshot
        page.screenshot(path="screenshots/initial.png", full_page=True)
        print("Initial screenshot taken")

        # Find and hover over suggested items
        suggested_items = page.locator(".suggested-item")
        count = suggested_items.count()
        print(f"Found {count} suggested items")

        if count > 0:
            # Get computed styles for first suggested item
            first_item = suggested_items.first

            # Normal state
            normal_bg = first_item.evaluate("el => window.getComputedStyle(el).backgroundColor")
            print(f"Normal background: {normal_bg}")

            # Hover state
            first_item.hover()
            time.sleep(0.5)
            hover_bg = first_item.evaluate("el => window.getComputedStyle(el).backgroundColor")
            print(f"Hover background: {hover_bg}")

            # Take screenshot while hovering
            page.screenshot(path="screenshots/hover_state.png", full_page=True)
            print("Hover screenshot taken")

            # Click on a suggested item
            first_item.click()
            time.sleep(1)
            page.screenshot(path="screenshots/after_click.png", full_page=True)
            print("After click screenshot taken")

        # Test New Chat button
        new_chat_btn = page.locator("#newChatButton")
        if new_chat_btn.count() > 0:
            print("Testing New Chat button...")

            # Add some messages first
            chat_input = page.locator("#chatInput")
            chat_input.fill("Test message")
            page.locator("#sendButton").click()
            time.sleep(2)

            # Click New Chat
            new_chat_btn.click()
            time.sleep(1)

            # Check if chat was cleared
            messages = page.locator(".message")
            print(f"Messages after New Chat: {messages.count()}")
            page.screenshot(path="screenshots/after_new_chat.png", full_page=True)

        # Test the test page
        page.goto("http://127.0.0.1:8000/test.html")
        page.wait_for_load_state("networkidle")

        test_items = page.locator(".suggested-item")
        if test_items.count() > 0:
            test_item = test_items.first

            # Normal state on test page
            normal_test_bg = test_item.evaluate("el => window.getComputedStyle(el).backgroundColor")
            print(f"Test page normal background: {normal_test_bg}")

            # Hover on test page
            test_item.hover()
            time.sleep(0.5)
            hover_test_bg = test_item.evaluate("el => window.getComputedStyle(el).backgroundColor")
            print(f"Test page hover background: {hover_test_bg}")

            page.screenshot(path="screenshots/test_page_hover.png", full_page=True)

        # Keep browser open for manual inspection
        input("Press Enter to close browser...")
        browser.close()

if __name__ == "__main__":
    import os
    os.makedirs("screenshots", exist_ok=True)
    test_rag_ui()