from playwright.sync_api import sync_playwright
import time
import json

def test_new_chat_detailed():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Enable console logging
        page.on("console", lambda msg: print(f"Console: {msg.text}"))

        # Go to page
        print("Loading page...")
        page.goto("http://127.0.0.1:8000")
        page.wait_for_load_state("networkidle")

        # Execute JavaScript to check button and function
        result = page.evaluate("""
            () => {
                const btn = document.getElementById('newChatButton');
                const funcExists = typeof startNewChat;
                const sessionId = currentSessionId;

                // Try to get the actual function code
                let funcCode = 'not found';
                if (typeof startNewChat === 'function') {
                    funcCode = startNewChat.toString().substring(0, 200);
                }

                return {
                    buttonExists: btn !== null,
                    buttonText: btn ? btn.textContent : null,
                    functionType: funcExists,
                    currentSession: sessionId,
                    functionCode: funcCode
                };
            }
        """)

        print("\nButton and Function Check:")
        print(json.dumps(result, indent=2))

        # Send a message first
        print("\nSending test message...")
        page.fill("#chatInput", "test message")
        page.click("#sendButton")
        time.sleep(3)

        # Get message count before
        messages_before = page.locator(".message").count()
        print(f"Messages before: {messages_before}")

        # Try to click New Chat button
        if result['buttonExists']:
            print("\nClicking New Chat button...")

            # Click and wait
            page.click("#newChatButton")
            time.sleep(2)

            # Get message count after
            messages_after = page.locator(".message").count()
            print(f"Messages after: {messages_after}")

            # Check session ID change
            new_session = page.evaluate("currentSessionId")
            print(f"Session after click: {new_session}")

            if messages_after < messages_before:
                print("SUCCESS: Messages were cleared!")
            else:
                print("ISSUE: Messages were not cleared")

                # Try calling function directly
                print("\nTrying to call startNewChat directly...")
                page.evaluate("startNewChat()")
                time.sleep(2)

                messages_direct = page.locator(".message").count()
                print(f"Messages after direct call: {messages_direct}")

        browser.close()

if __name__ == "__main__":
    test_new_chat_detailed()