from playwright.sync_api import sync_playwright
import time

def test_console_errors():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Show browser
        page = browser.new_page()

        # Capture console messages
        console_messages = []
        page.on("console", lambda msg: console_messages.append(f"{msg.type}: {msg.text}"))

        # Capture page errors
        page_errors = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # Go to the main page
        print("Loading page...")
        page.goto("http://127.0.0.1:8000")
        page.wait_for_load_state("networkidle")

        # Print any console messages
        if console_messages:
            print("\nConsole messages:")
            for msg in console_messages:
                print(f"  {msg}")

        if page_errors:
            print("\nPage errors:")
            for err in page_errors:
                print(f"  {err}")

        # Check if newChatButton exists
        new_chat_btn = page.locator("#newChatButton")
        if new_chat_btn.count() > 0:
            print(f"\n✓ New Chat button found")

            # Check if it has click handler
            has_handler = page.evaluate("""
                () => {
                    const btn = document.getElementById('newChatButton');
                    if (!btn) return 'Button not found';

                    // Check for event listeners (this is limited, but can help)
                    const events = getEventListeners ? getEventListeners(btn) : null;

                    // Try to check if startNewChat function exists
                    const funcExists = typeof startNewChat === 'function';

                    return {
                        buttonFound: true,
                        startNewChatExists: funcExists,
                        onclick: btn.onclick ? 'Has onclick' : 'No onclick'
                    };
                }
            """)
            print(f"Button check: {has_handler}")

            # Try clicking the button
            print("\nClicking New Chat button...")
            new_chat_btn.click()
            time.sleep(1)

            # Check for new console messages after click
            if len(console_messages) > 0:
                print("Console after click:")
                for msg in console_messages[-5:]:  # Last 5 messages
                    print(f"  {msg}")
        else:
            print("\n✗ New Chat button NOT found!")

        # Test if startNewChat function is accessible
        func_check = page.evaluate("typeof startNewChat")
        print(f"\nstartNewChat function type: {func_check}")

        # Keep browser open for inspection
        print("\nBrowser will stay open. Press Enter to close...")
        input()
        browser.close()

if __name__ == "__main__":
    test_console_errors()