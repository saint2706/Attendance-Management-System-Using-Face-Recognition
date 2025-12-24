from playwright.sync_api import sync_playwright
import time

def verify_accessibility_attributes():
    with sync_playwright() as p:
        # Launch browser with fake media stream args
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream"
            ]
        )

        # Create context blocking service workers to ensure network requests are intercepted if needed
        # Grant camera permissions
        context = browser.new_context(
            service_workers='block',
            permissions=["camera"]
        )
        page = context.new_page()

        # Prevent video.play() from failing
        page.add_init_script("""
            const originalPlay = HTMLMediaElement.prototype.play;
            HTMLMediaElement.prototype.play = async function() {
                // If it's our camera video, just pretend to play
                if (this.classList.contains('camera-video')) {
                    console.log("Mocking video.play()");
                    return Promise.resolve();
                }
                return originalPlay.call(this);
            };
        """)

        # Mock API response for attendance
        page.route("**/api/attendance/**", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"recognized": true, "username": "Test User", "confidence": 0.95}'
        ))

        # Navigate to the page
        print("Navigating to page...")
        page.goto("http://localhost:5173/static/mark-attendance")

        try:
            # 1. Check loader role="status"
            # It might appear briefly.
            print("Checking for loader...")
            loader = page.locator("div[role='status']")
            try:
                if loader.is_visible():
                    print("✅ Found loader with role='status'")
                else:
                    print("ℹ️ Loader not visible (probably finished initializing).")
            except:
                print("ℹ️ Error checking loader.")

            # 2. Check video attributes
            print("Waiting for video element...")
            video = page.locator("video.camera-video")
            # Wait for it to be visible. If camera works, it should be visible.
            video.wait_for(state="visible", timeout=10000)
            print("Video visible.")

            aria_describedby = video.get_attribute("aria-describedby")
            if aria_describedby == "attendance-instructions":
                print("✅ Video has aria-describedby='attendance-instructions'")
            else:
                print(f"❌ Video missing correct aria-describedby. Found: {aria_describedby}")

            # 3. Check instructions ID
            instructions = page.locator("#attendance-instructions")
            if instructions.count() > 0:
                print("✅ Instructions card has id='attendance-instructions'")
            else:
                print("❌ Instructions card missing id='attendance-instructions'")

            # 4. Check face guide aria-hidden
            face_guide = page.locator(".face-guide")
            aria_hidden = face_guide.get_attribute("aria-hidden")
            if aria_hidden == "true":
                print("✅ Face guide has aria-hidden='true'")
            else:
                print(f"❌ Face guide missing aria-hidden='true'. Found: {aria_hidden}")

            # 5. Trigger Capture to check Result Card role="alert"
            print("Clicking Capture button...")
            capture_btn = page.locator("button.capture-button")
            # Force click even if disabled check fails (though it should be enabled)
            capture_btn.click()

            print("Waiting for result card...")
            result_card = page.locator(".result-card")
            result_card.wait_for(state="visible", timeout=5000)

            role = result_card.get_attribute("role")
            if role == "alert":
                print("✅ Result card has role='alert'")
            else:
                print(f"❌ Result card missing role='alert'. Found: {role}")

            aria_live = result_card.get_attribute("aria-live")
            if aria_live == "assertive":
                print("✅ Result card has aria-live='assertive'")
            else:
                print(f"❌ Result card missing aria-live='assertive'. Found: {aria_live}")

            # Take screenshot
            page.screenshot(path="verification/accessibility_verified.png")
            print("Screenshot saved to verification/accessibility_verified.png")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="verification/error_retry_3.png")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_accessibility_attributes()
