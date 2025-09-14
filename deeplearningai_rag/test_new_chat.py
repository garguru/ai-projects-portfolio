from playwright.sync_api import sync_playwright
import time

def test_new_chat():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Go to the main page
        page.goto("http://127.0.0.1:8000")
        page.wait_for_load_state("networkidle")

        # Send a test message
        chat_input = page.locator("#chatInput")
        chat_input.fill("Test message before new chat")
        page.locator("#sendButton").click()

        # Wait for response
        time.sleep(3)

        # Count messages before new chat
        messages_before = page.locator(".message").count()
        print(f"Messages before New Chat: {messages_before}")

        # Take screenshot before
        page.screenshot(path="screenshots/before_new_chat.png")

        # Click New Chat button
        new_chat_btn = page.locator("#newChatButton")
        if new_chat_btn.count() > 0:
            print("Clicking New Chat button...")
            new_chat_btn.click()
            time.sleep(1)

            # Count messages after new chat
            messages_after = page.locator(".message").count()
            print(f"Messages after New Chat: {messages_after}")

            # Take screenshot after
            page.screenshot(path="screenshots/after_new_chat.png")

            if messages_after == 0:
                print("✅ New Chat button works! Chat was cleared.")
            else:
                print(f"❌ New Chat button issue: Still {messages_after} messages")
        else:
            print("❌ New Chat button not found")

        browser.close()

if __name__ == "__main__":
    test_new_chat()