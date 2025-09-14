from playwright.sync_api import sync_playwright
import time
import os

def test_rag_ui():
    os.makedirs("screenshots", exist_ok=True)

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
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

            # Get the actual CSS content being used
            css_link = page.locator('link[rel="stylesheet"][href*="style.css"]').first
            css_href = css_link.get_attribute("href")
            print(f"CSS file being used: {css_href}")

            # Check what styles are actually applied
            first_item_styles = first_item.evaluate("""el => {
                const styles = window.getComputedStyle(el);
                return {
                    background: styles.backgroundColor,
                    border: styles.borderColor,
                    color: styles.color
                }
            }""")
            print(f"First item computed styles: {first_item_styles}")

        # Check if rgb(217, 199, 178) appears anywhere (that's the tan color)
        tan_color = "rgb(217, 199, 178)"
        if tan_color in hover_bg:
            print(f"WARNING: Tan color detected in hover state! {hover_bg}")
        else:
            print(f"Good: Hover color is {hover_bg}, not tan")

        browser.close()
        print("\nScreenshots saved in screenshots/ folder")

if __name__ == "__main__":
    test_rag_ui()