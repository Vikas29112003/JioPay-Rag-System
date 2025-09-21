#!/usr/bin/env python3
"""
Enhanced JioPay FAQ Scraper
Based on successful simple scraper approach - extracts FAQs from ALL categories systematically
"""

import asyncio
import json
import logging
from datetime import datetime
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedJioPayFAQScraper:
    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None
        self.faqs = []
        
    async def setup(self):
        """Setup browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        
    async def navigate_to_help_center(self):
        """Navigate to help center"""
        await self.page.goto("https://jiopay.com/business/help-center")
        await asyncio.sleep(3)
        
    async def extract_text_after_click(self, locator, question_text, category):
        """Extract answer text after clicking an element - using successful approach"""
        try:
            # Click the element
            await locator.click()
            await asyncio.sleep(2)
            
            # Get page content
            content = await self.page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Strategy 1: Look for answer patterns
            answer_text = ""
            
            # Try to find the answer after the question
            all_text = soup.get_text()
            if question_text in all_text:
                question_index = all_text.find(question_text)
                text_after = all_text[question_index + len(question_text):question_index + len(question_text) + 2000]
                
                # Split into lines and filter
                lines = [line.strip() for line in text_after.split('\n') if line.strip()]
                
                # Get meaningful lines (skip navigation/repeated content)
                for i, line in enumerate(lines):
                    if len(line) > 30 and not any(skip in line.lower() for skip in ['menu', 'navigation', 'footer', 'header']):
                        answer_text = line
                        break
            
            # Strategy 2: Look for specific answer elements
            if not answer_text or len(answer_text) < 30:
                # Look for divs with substantial content
                content_divs = soup.find_all(['div', 'p', 'section'])
                for div in content_divs:
                    text = div.get_text(strip=True)
                    if (text and len(text) > 50 and 
                        question_text.lower() not in text.lower() and
                        any(word in text.lower() for word in ['app', 'business', 'jiopay', 'payment', 'dashboard'])):
                        
                        # Check if this looks like an answer
                        if ('.' in text or 'steps' in text.lower() or 'follow' in text.lower() or 
                            'contact' in text.lower() or 'ensure' in text.lower()):
                            answer_text = text
                            break
            
            if answer_text and len(answer_text.strip()) > 30:
                faq_entry = {
                    "category": category,
                    "question": question_text,
                    "answer": answer_text.strip(),
                    "timestamp": datetime.now().isoformat()
                }
                self.faqs.append(faq_entry)
                logger.info(f"✅ Extracted FAQ: {question_text[:50]}...")
                return True
            else:
                logger.warning(f"❌ No substantial answer found for: {question_text}")
                return False
                
        except Exception as e:
            logger.error(f"Error clicking {question_text}: {e}")
            return False
    
    async def process_category(self, category_name, questions_list):
        """Process a specific FAQ category"""
        logger.info(f"🔄 Processing category: {category_name}")
        
        try:
            # Click category button
            category_button = self.page.get_by_role("button", name=category_name)
            if await category_button.count() > 0:
                await category_button.click()
                await asyncio.sleep(2)
                logger.info(f"📂 Opened category: {category_name}")
            else:
                logger.warning(f"❌ Category button not found: {category_name}")
                return
            
            # Process each question in the category
            extracted_count = 0
            for question in questions_list:
                try:
                    # Try exact match first
                    question_locator = self.page.get_by_text(question, exact=True)
                    
                    if await question_locator.count() > 0:
                        success = await self.extract_text_after_click(question_locator.first, question, category_name)
                        if success:
                            extracted_count += 1
                    else:
                        # Try partial match
                        partial_question = question[:40] if len(question) > 40 else question
                        partial_locator = self.page.get_by_text(partial_question)
                        
                        if await partial_locator.count() > 0:
                            success = await self.extract_text_after_click(partial_locator.first, question, category_name)
                            if success:
                                extracted_count += 1
                        else:
                            logger.warning(f"❌ Question not found: {question[:50]}...")
                    
                    await asyncio.sleep(1)  # Brief pause between questions
                    
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {e}")
                    continue
            
            logger.info(f"✅ Category {category_name}: Extracted {extracted_count}/{len(questions_list)} FAQs")
            
        except Exception as e:
            logger.error(f"Error processing category {category_name}: {e}")
    
    async def scrape_all_faqs(self):
        """Scrape all FAQ categories systematically"""
        
        # Define all categories and their questions
        faq_categories = {
            "JioPay Business App": [
                "What is JioPay Business?",
                "What is the purpose of the JioPay Business App?",
                "How can I download the JioPay Business App?",
                "I have forgotten my account password, how can I reset it?",
                "I am unable to login to the app, what should I do?",
                "Why My App is crashing on my Phone?",
                "Where can I see transaction summary in the app?"
            ],
            
            "JioPay Business Dashboard": [
                "What is JioPay Business Dashboard?",
                "How can I generate reports on JioPay business Dashboard?"
            ],
            
            "Collect link": [
                "How can I create Collect link?",
                "What are the payment modes available via Collect link?",
                "Can I use Single Collect link to accept payments from multiple customers?",
                "What is the validity of the Collect link?",
                "Can I create Bulk Collect links?",
                "Is partial payment allowed?",
                "Can customer enter the amount?"
            ],
            
            "User Management": [
                "Can I add sub user to JioPay Business?",
                "How can a new sub access merchant Dashboard?",
                "Can I block sub user?"
            ],
            
            "Repeat": [
                "What is Repeat?",
                "What are the payment methods supported for Repeat?",
                "What is the maximum amount for debit without 2FA in subsequent payment?",
                "I want to give my customer a free trial, would that be possible?",
                "Can I create Repeat via dashboard?",
                "Will you be able to manage my subscriptions?"
            ],
            
            "Campaign": [
                "How can I create campaign?",
                "How can I edit campaign?",
                "How can I pause/stop campaign?"
            ],
            
            "Settlement": [
                "What are settlements?",
                "How to check settlements in my bank account?",
                "What should I do if I'm not receiving my settlements?",
                "I believe that I have received partial or incorrect settlement in my account?",
                "How do I Update settlement bank account number?",
                "Do I have to do manual settlement for my account every day?",
                "Why is my settlement on hold for some transactions?"
            ],
            
            "Refunds": [
                "How can I issue refunds to my customers for any payments made by them?",
                "How to check the status of refund?",
                "How to check ARN for refund?",
                "What should I do if refund is not credited in my customer's account?",
                "Can I cancel a refund?",
                "Do you charge for refund?",
                "Can we do bulk refund?",
                "What are the steps for Bulk refund?",
                "Do we have a format for bulk refund report?",
                "Is partial refund allowed in bulk refund?",
                "Can we reprocess failed record in bulk refund?"
            ],
            
            "Notifications": [
                "How can I disable SMS notification from dashboard?",
                "How can I add new number for SMS from dashboard?"
            ],
            
            "Voicebox": [
                "What is the JioPay VoiceBox?",
                "How does the VoiceBox work?",
                "How does JioPay VoiceBox compare with other devices?",
                "How do I get a new VoiceBox?",
                "Is doorstep installation included with the JioPay VoiceBox?",
                "How can I set up the JioPay VoiceBox?",
                "Can I use any SIM in the VoiceBox?",
                "What if I would like to return /replace the VoiceBox?",
                "Can the JioPay VoiceBox be used in noisy environments?",
                "What are some measures to take to keep the voice box in good working condition?",
                "What type of transactions will VoiceBox announce?",
                "What type of transactions can be supported/voiced out?",
                "What type of languages are supported for announcements?",
                "How can I change the language of announcements?",
                "How do I replay the last transaction on the VoiceBox?",
                "My VoiceBox is not working, what should I do?",
                "What if my VoiceBox is not charging?",
                "How do I power on my VoiceBox and verify it is operational?",
                "What if the device is not turning on?",
                "What if device is not getting connected to network?",
                "What are the charges for the VoiceBox?",
                "How can I get an invoice for the payment made?",
                "How do I control the volume of the JioPay VoiceBox?",
                "How do I check the battery level of the JioPay VoiceBox?"
            ],
            
            "DQR": [
                "What should a store manager do on receipt of JioPay DQR standee?",
                "Who will send the store manager the JioPay DQR?",
                "What if the neighborhood smart point have it and a particular store manager doesn't have it?",
                "What if the DQR device is defective?",
                "What if the store manager has received excess or lesser number of devices?",
                "How do I start using the DQR device for transactions?",
                "What all UPI payment applications/options would JioPay DQR support?",
                "What if the DQR is not working after connecting to RPoS Billing System?",
                "How to initiate refund in normal DQR transactions",
                "Will there be any training provided on the usage of DQR?",
                "In case of transaction timeout how to check if money is credited or not?",
                "How will the settlement happen in case of payment made by customer via DQR?",
                "When to use \"Cancel\" option in check status?",
                "When not to use \"Cancel\" option? Scenarios where cancel option should not be used"
            ],
            
            "Partner program": [
                "Why should you consider becoming a part of the JioPay Business Partner program?",
                "What is the potential earning structure within the JioPay Business Partner Program?",
                "Can a business that's already registered with JioPay Business also sign up as a partner?"
            ],
            
            "P2PM / Low KYC merchants": [
                "Who are P2PM Merchants?",
                "What are Limitations of being a P2PM Merchant?",
                "What are benefits of becoming a P2M merchant?",
                "How long would it require to become P2M merchant after upgradation request?",
                "What if a P2PM Merchant merchants breaches ₹ 1,00,000/- monthly limit?",
                "How to change the settlement account?"
            ]
        }
        
        # Calculate total questions
        total_questions = sum(len(questions) for questions in faq_categories.values())
        logger.info(f"🎯 Target: {total_questions} FAQ questions across {len(faq_categories)} categories")
        
        # Process each category
        for category_name, questions in faq_categories.items():
            await self.process_category(category_name, questions)
            await asyncio.sleep(2)  # Pause between categories
    
    async def save_faqs(self):
        """Save extracted FAQs to JSON file"""
        if not self.faqs:
            logger.warning("No FAQs extracted!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Scraped Data/jiopay_enhanced_faqs.json"
        
        # Group FAQs by category for better organization
        categories = {}
        for faq in self.faqs:
            category = faq['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(faq)
        
        output_data = {
            "extraction_info": {
                "timestamp": datetime.now().isoformat(),
                "total_faqs": len(self.faqs),
                "categories": list(categories.keys()),
                "category_counts": {cat: len(faqs) for cat, faqs in categories.items()},
                "method": "enhanced_systematic_extraction"
            },
            "faqs": self.faqs,
            "categories": categories
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved {len(self.faqs)} FAQs to {filename}")
            
            # Print detailed summary
            print(f"\n📊 EXTRACTION SUMMARY:")
            print(f"Total FAQs Extracted: {len(self.faqs)}")
            print(f"Categories Processed: {len(categories)}")
            print(f"\nBy Category:")
            for category, count in output_data['extraction_info']['category_counts'].items():
                print(f"  {category}: {count} FAQs")
                
        except Exception as e:
            logger.error(f"Error saving FAQs: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()

async def main():
    scraper = EnhancedJioPayFAQScraper()
    
    try:
        logger.info("🚀 Starting Enhanced JioPay FAQ extraction...")
        await scraper.setup()
        await scraper.navigate_to_help_center()
        await scraper.scrape_all_faqs()
        await scraper.save_faqs()
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
    finally:
        await scraper.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
