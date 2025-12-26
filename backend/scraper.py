from playwright.sync_api import sync_playwright

PORTFOLIO_URL = "http://localhost:5173"

def scrape_portfolio():
    documents = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(PORTFOLIO_URL)

        # Wait for React to render
        page.wait_for_selector("section")

        sections = page.query_selector_all("section")

        for section in sections:
            section_id = section.get_attribute("id") or "unknown"
            text = section.inner_text().strip()

            if len(text) < 50:
                continue

            documents.append({
                "content": text,
                "metadata": {
                    "section": section_id
                }
            })

        browser.close()

    return documents
