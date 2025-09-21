#!/usr/bin/env python3
"""
Enhanced JioPay Website Scraper using Selenium
Clicks through all clickable elements and navigates through all pages
to extract comprehensive content including About Us, FAQs, and business info
Similar functionality to the Playwright version but using Selenium
"""

import json
import time
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedJioPaySeleniumScraper:
    def __init__(self, base_url="https://jiopay.com"):
        self.base_url = base_url
        self.business_url = f"{base_url}/business"
        self.scraped_data = {
            "site_info": {
                "base_url": self.base_url,
                "business_url": self.business_url,
                "scrape_timestamp": datetime.now().isoformat(),
                "total_pages_scraped": 0,
                "total_clicks_performed": 0
            },
            "pages": [],
            "navigation": [],
            "about_us": {},
            "faqs": [],
            "business_info": {},
            "all_links": [],
            "content_sections": [],
            "clicked_elements": [],
            "modal_content": [],
            "form_data": []
        }
        self.visited_urls = set()
        self.clicked_elements = set()
        self.driver = None
        
    def setup_driver(self):
        """Set up Chrome driver with appropriate options"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Run in background
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            logger.info("Chrome driver setup successful")
            return True
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            return False
    
    def wait_for_page_load(self, timeout=30):
        """Wait for page to fully load"""
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            # Additional wait for dynamic content
            time.sleep(3)
            # Scroll to load more content if needed
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        except TimeoutException:
            logger.warning("Page load timeout, continuing anyway")
    
    def click_element_safely(self, element, element_info):
        """Safely click an element with error handling"""
        try:
            # Scroll element into view
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
            time.sleep(0.5)
            
            # Check if element is clickable
            if element.is_displayed() and element.is_enabled():
                # Try different click methods
                try:
                    element.click()
                except ElementNotInteractableException:
                    # Try JavaScript click if regular click fails
                    self.driver.execute_script("arguments[0].click();", element)
                except Exception:
                    # Try ActionChains click as last resort
                    ActionChains(self.driver).move_to_element(element).click().perform()
                
                time.sleep(1)
                self.scraped_data["site_info"]["total_clicks_performed"] += 1
                
                # Record click information
                click_info = {
                    "element_type": element_info.get("tag_name", "unknown"),
                    "text": element_info.get("text", ""),
                    "selector": element_info.get("selector", ""),
                    "timestamp": datetime.now().isoformat(),
                    "url_before_click": self.driver.current_url,
                    "url_after_click": None
                }
                
                # Wait for potential navigation or content change
                try:
                    WebDriverWait(self.driver, 5).until(
                        lambda driver: driver.execute_script("return document.readyState") == "complete"
                    )
                except TimeoutException:
                    pass
                
                click_info["url_after_click"] = self.driver.current_url
                self.scraped_data["clicked_elements"].append(click_info)
                
                logger.info(f"Clicked element: {element_info.get('text', 'No text')[:50]}")
                return True
        except Exception as e:
            logger.warning(f"Failed to click element {element_info.get('text', '')[:30]}: {e}")
            return False
    
    def extract_page_content(self, url):
        """Extract comprehensive content from current page"""
        try:
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'lxml')
            
            page_data = {
                "url": url,
                "title": self.driver.title,
                "timestamp": datetime.now().isoformat(),
                "meta_description": "",
                "navigation_links": [],
                "all_links": [],
                "about_us": {},
                "faqs": [],
                "business_info": {},
                "content_sections": [],
                "forms": [],
                "modals": [],
                "full_text": soup.get_text(strip=True, separator=' ')
            }
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                page_data["meta_description"] = meta_desc.get('content', '')
            
            # Extract navigation links
            page_data["navigation_links"] = self.extract_navigation_links(soup, url)
            
            # Extract all links
            page_data["all_links"] = self.extract_links(soup, url)
            
            # Extract About Us content
            page_data["about_us"] = self.extract_about_us(soup)
            
            # Extract FAQs
            page_data["faqs"] = self.extract_faqs(soup)
            
            # Extract business information
            page_data["business_info"] = self.extract_business_info(soup)
            
            # Extract content sections
            page_data["content_sections"] = self.extract_content_sections(soup)
            
            # Extract forms
            page_data["forms"] = self.extract_forms(soup)
            
            return page_data
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def extract_navigation_links(self, soup, url):
        """Extract navigation links from the page"""
        nav_links = []
        nav_selectors = ['nav', '.nav', '.navigation', '.menu', '.header-menu', 'header nav']
        
        for selector in nav_selectors:
            nav_elements = soup.select(selector)
            for nav in nav_elements:
                links = nav.find_all('a', href=True)
                for link in links:
                    href = link.get('href')
                    text = link.get_text(strip=True)
                    if href and text:
                        full_url = urljoin(url, href)
                        nav_links.append({
                            "text": text,
                            "url": full_url,
                            "is_internal": self.is_internal_link(full_url)
                        })
        
        return nav_links
    
    def extract_links(self, soup, base_url):
        """Extract all links from the page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text(strip=True)
            if href:
                full_url = urljoin(base_url, href)
                links.append({
                    "text": text,
                    "url": full_url,
                    "is_internal": self.is_internal_link(full_url)
                })
        return links
    
    def is_internal_link(self, url):
        """Check if a link is internal to the domain"""
        try:
            parsed_url = urlparse(url)
            base_domain = urlparse(self.base_url).netloc
            return parsed_url.netloc == base_domain or parsed_url.netloc == ''
        except:
            return False
    
    def extract_about_us(self, soup):
        """Extract About Us information"""
        about_patterns = [
            'about', 'about-us', 'about_us', 'company', 'who-we-are',
            'our-story', 'mission', 'vision', 'values'
        ]
        
        about_content = {}
        
        # Look for sections with about-related content
        for pattern in about_patterns:
            # Look by ID
            element = soup.find(id=re.compile(pattern, re.I))
            if element:
                about_content[pattern] = {
                    "heading": self.get_heading_text(element),
                    "content": element.get_text(strip=True, separator=' ')
                }
            
            # Look by class
            elements = soup.find_all(class_=re.compile(pattern, re.I))
            for elem in elements:
                about_content[f"{pattern}_class"] = {
                    "heading": self.get_heading_text(elem),
                    "content": elem.get_text(strip=True, separator=' ')
                }
        
        return about_content
    
    
    def extract_faqs(self, soup):
        """Extract FAQ information with enhanced detection"""
        faqs = []
        
        # Look for FAQ patterns in text and structure
        faq_patterns = ['faq', 'frequently asked', 'questions', 'help', 'support']
        
        for pattern in faq_patterns:
            # Find FAQ containers
            faq_containers = soup.find_all(string=re.compile(pattern, re.I))
            
            for container_text in faq_containers:
                if hasattr(container_text, 'parent'):
                    container = container_text.parent
                    
                    # Look for question-answer structure
                    questions = container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    
                    for question in questions:
                        q_text = question.get_text(strip=True)
                        
                        # Look for answer in next siblings
                        answer_elem = question.find_next_sibling(['p', 'div', 'span'])
                        if not answer_elem:
                            answer_elem = question.find_next(['p', 'div'])
                        
                        if answer_elem:
                            a_text = answer_elem.get_text(strip=True)
                            if q_text and a_text and len(a_text) > 10:
                                faqs.append({
                                    "question": q_text,
                                    "answer": a_text
                                })
        
        return faqs
    
    def extract_business_info(self, soup):
        """Extract business-specific information"""
        business_info = {}
        
        business_patterns = [
            'business', 'enterprise', 'corporate', 'merchant', 'solution',
            'service', 'feature', 'benefit', 'product', 'offering'
        ]
        
        for pattern in business_patterns:
            elements = soup.find_all(string=re.compile(pattern, re.I))
            
            for elem in elements:
                if hasattr(elem, 'parent') and elem.parent:
                    parent = elem.parent
                    
                    # Get surrounding content
                    content_block = parent.find_parent(['div', 'section', 'article'])
                    if content_block:
                        heading = self.get_heading_text(content_block)
                        content = content_block.get_text(strip=True, separator=' ')
                        
                        if content and len(content) > 50:
                            business_info[f"{pattern}_{len(business_info)}"] = {
                                "heading": heading or f"{pattern.title()} Information",
                                "content": content
                            }
        
        return business_info
    
    def extract_forms(self, soup):
        """Extract form information"""
        forms = []
        
        for form in soup.find_all('form'):
            form_data = {
                "action": form.get('action', ''),
                "method": form.get('method', 'GET'),
                "fields": []
            }
            
            # Extract form fields
            for field in form.find_all(['input', 'select', 'textarea']):
                field_info = {
                    "type": field.get('type', field.name),
                    "name": field.get('name', ''),
                    "placeholder": field.get('placeholder', ''),
                    "required": field.has_attr('required')
                }
                form_data["fields"].append(field_info)
            
            if form_data["fields"]:
                forms.append(form_data)
        
        return forms
    
    def find_clickable_elements(self):
        """Find all clickable elements on the page"""
        clickable_elements = []
        
        # Selectors for clickable elements
        selectors = [
            'a[href]',
            'button',
            '[onclick]',
            '[role="button"]',
            '[role="tab"]',
            '[role="menuitem"]',
            '.btn',
            '.button',
            '.link',
            '.menu-item',
            '.nav-link',
            'input[type="submit"]',
            'input[type="button"]',
            '[data-toggle]',
            '[data-target]'
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    try:
                        # Get element information
                        tag_name = element.tag_name.lower()
                        text = element.text.strip()
                        href = element.get_attribute('href')
                        
                        # Create unique identifier
                        element_id = f"{tag_name}_{text[:30]}_{href}"
                        
                        if element_id not in self.clicked_elements:
                            element_info = {
                                "tag_name": tag_name,
                                "text": text,
                                "href": href,
                                "selector": selector,
                                "element_id": element_id
                            }
                            
                            clickable_elements.append((element, element_info))
                            
                    except Exception as e:
                        logger.debug(f"Error processing element: {e}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Error with selector {selector}: {e}")
                continue
        
        return clickable_elements
    
    def click_through_page(self, url):
        """Click through all elements on a page and extract content"""
        logger.info(f"Processing page: {url}")
        
        self.driver.get(url)
        self.wait_for_page_load()
        
        # Extract initial content
        page_data = self.extract_page_content(url)
        if page_data:
            self.scraped_data["pages"].append(page_data)
        
        # Find and click all clickable elements
        clickable_elements = self.find_clickable_elements()
        
        logger.info(f"Found {len(clickable_elements)} clickable elements")
        
        for element, element_info in clickable_elements:
            try:
                # Skip external links and already visited pages
                href = element_info.get("href")
                if href and not self.is_internal_link(href):
                    continue
                
                current_url = self.driver.current_url
                self.click_element_safely(element, element_info)
                
                # Check if we're on a new page or modal opened
                new_url = self.driver.current_url
                
                if new_url != current_url and new_url not in self.visited_urls:
                    # New page loaded
                    self.visited_urls.add(new_url)
                    self.wait_for_page_load()
                    
                    new_page_data = self.extract_page_content(new_url)
                    if new_page_data:
                        self.scraped_data["pages"].append(new_page_data)
                        logger.info(f"Discovered new page: {new_url}")
                    
                    # Go back to original page
                    self.driver.get(current_url)
                    self.wait_for_page_load()
                else:
                    # Possible modal or dynamic content change
                    modal_content = self.extract_modal_content()
                    if modal_content:
                        self.scraped_data["modal_content"].append(modal_content)
                
                # Mark element as clicked
                self.clicked_elements.add(element_info["element_id"])
                
            except Exception as e:
                logger.warning(f"Error processing clickable element: {e}")
                continue
    
    def extract_modal_content(self):
        """Extract content from modals or overlays"""
        try:
            # Look for modal-like elements
            modal_selectors = [
                '[role="dialog"]',
                '.modal',
                '.popup',
                '.overlay',
                '[aria-modal="true"]'
            ]
            
            for selector in modal_selectors:
                try:
                    modal_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if modal_element.is_displayed():
                        text_content = modal_element.text
                        if text_content and text_content.strip():
                            return {
                                "selector": selector,
                                "content": text_content.strip(),
                                "timestamp": datetime.now().isoformat(),
                                "url": self.driver.current_url
                            }
                except NoSuchElementException:
                    continue
            
            return None
        except Exception as e:
            logger.debug(f"Error extracting modal content: {e}")
            return None
    
    def discover_all_pages(self):
        """Discover all pages on the website"""
        pages_to_visit = [self.business_url, self.base_url]
        visited = set()
        
        # Start with main pages
        for start_url in pages_to_visit:
            if start_url not in visited:
                visited.add(start_url)
                self.click_through_page(start_url)
        
        # Process additional discovered pages
        discovered_pages = set()
        for page_data in self.scraped_data["pages"]:
            for link in page_data.get("all_links", []):
                if link["is_internal"] and link["url"] not in visited:
                    discovered_pages.add(link["url"])
        
        # Visit discovered pages (limit to prevent infinite crawling)
        for url in list(discovered_pages)[:15]:  # Limit to 15 additional pages
            if url not in visited:
                visited.add(url)
                try:
                    self.click_through_page(url)
                except Exception as e:
                    logger.error(f"Error processing discovered page {url}: {e}")
                    continue
    
    def aggregate_data(self):
        """Aggregate data from all pages"""
        all_navigation = []
        all_about_us = {}
        all_faqs = []
        all_business_info = {}
        all_links = []
        all_content_sections = []
        
        for page_data in self.scraped_data["pages"]:
            all_navigation.extend(page_data.get("navigation_links", []))
            all_about_us.update(page_data.get("about_us", {}))
            all_faqs.extend(page_data.get("faqs", []))
            all_business_info.update(page_data.get("business_info", {}))
            all_links.extend(page_data.get("all_links", []))
            all_content_sections.extend(page_data.get("content_sections", []))
        
        # Remove duplicates
        self.scraped_data["navigation"] = self.remove_duplicate_links(all_navigation)
        self.scraped_data["about_us"] = all_about_us
        self.scraped_data["faqs"] = self.remove_duplicate_faqs(all_faqs)
        self.scraped_data["business_info"] = all_business_info
        self.scraped_data["all_links"] = self.remove_duplicate_links(all_links)
        self.scraped_data["content_sections"] = all_content_sections
    
    def remove_duplicate_links(self, links):
        """Remove duplicate links"""
        seen = set()
        unique_links = []
        for link in links:
            link_key = link.get("url", "")
            if link_key not in seen:
                seen.add(link_key)
                unique_links.append(link)
        return unique_links
    
    def remove_duplicate_faqs(self, faqs):
        """Remove duplicate FAQs"""
        seen = set()
        unique_faqs = []
        for faq in faqs:
            faq_key = (faq.get("question", ""), faq.get("answer", ""))
            if faq_key not in seen:
                seen.add(faq_key)
                unique_faqs.append(faq)
        return unique_faqs
    
    def get_heading_text(self, element):
        """Extract heading text from an element"""
        heading = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        return heading.get_text(strip=True) if heading else ""
    
    def extract_content_sections(self, soup):
        """Extract all major content sections"""
        sections = []
        
        # Main content containers
        main_selectors = ['main', '.main', '.content', '.container', '.wrapper']
        
        for selector in main_selectors:
            containers = soup.select(selector)
            for container in containers:
                # Extract sections within containers
                section_elements = container.find_all(['section', 'div', 'article'])
                
                for section in section_elements:
                    heading = self.get_heading_text(section)
                    content = section.get_text(strip=True, separator=' ')
                    
                    if content and len(content) > 50:  # Filter out small/empty sections
                        sections.append({
                            "heading": heading or "Untitled Section",
                            "content": content[:1000] + "..." if len(content) > 1000 else content,
                            "full_content": content
                        })
        
        return sections
    
    
    def scrape_all(self):
        """Main scraping method"""
        if not self.setup_driver():
            logger.error("Failed to setup driver")
            return None
        
        try:
            # Process main pages including help center for FAQs
            pages_to_process = [
                self.business_url,
                self.base_url,
                f"{self.business_url}/help-center"  # Added help center for FAQs
            ]
            
            for url in pages_to_process:
                if url not in self.visited_urls:
                    self.visited_urls.add(url)
                    self.click_through_page(url)
            
            # Update final stats
            self.scraped_data["site_info"]["total_pages_scraped"] = len(self.scraped_data["pages"])
            self.scraped_data["site_info"]["completion_timestamp"] = datetime.now().isoformat()
            
            return self.scraped_data
            
        except Exception as e:
            logger.error(f"Error in main scraping process: {e}")
            return self.scraped_data
        finally:
            if self.driver:
                self.driver.quit()
    
    def save_to_json(self, filename=None):
        """Save scraped data to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"../Scraped Data/jiopay_selenium_data.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Data saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            return None

def main():
    """Main execution function"""
    logger.info("Starting comprehensive JioPay website scraping with Enhanced Selenium...")
    
    scraper = EnhancedJioPaySeleniumScraper()
    scraped_data = scraper.scrape_all()
    
    if scraped_data:
        # Save to JSON file
        output_file = scraper.save_to_json()
        
        if output_file:
            print(f"\n✅ Comprehensive Selenium scraping completed successfully!")
            print(f"📁 Data saved to: {output_file}")
            print(f"📊 Total pages scraped: {scraped_data['site_info']['total_pages_scraped']}")
            print(f"🖱️  Total clicks performed: {scraped_data['site_info']['total_clicks_performed']}")
            print(f"🔗 Total links found: {len(scraped_data['all_links'])}")
            print(f"❓ Total FAQs found: {len(scraped_data['faqs'])}")
            print(f"📝 Total content sections: {len(scraped_data['content_sections'])}")
            print(f"🖼️  Modal content captured: {len(scraped_data['modal_content'])}")
            print(f"📋 Clicked elements logged: {len(scraped_data['clicked_elements'])}")
        else:
            print("❌ Error saving data to file")
    else:
        print("❌ Scraping failed")

if __name__ == "__main__":
    main()
