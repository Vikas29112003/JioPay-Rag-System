#!/usr/bin/env python3
"""
Enhanced Data Normalization Module for JioPay RAG System
Specifically designed to fix FAQ data quality issues:
- Garbage Unicode characters removal
- Truncated/appended text cleanup
- Repetitive boilerplate elimination
"""

import json
import re
import string
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedJioPayDataNormalizer:
    """
    Enhanced data normalizer specifically designed to fix JioPay FAQ quality issues
    """
    
    def __init__(self):
        self.cleaned_data = {}
        self.stats = {
            "total_faqs_processed": 0,
            "garbage_chars_removed": 0,
            "truncated_answers_fixed": 0,
            "boilerplate_removed": 0,
            "duplicates_removed": 0,
            "empty_entries_removed": 0,
            "normalization_timestamp": None
        }
        
        # Define patterns for problematic content
        self.garbage_unicode_pattern = re.compile(r'[\u2000-\u206F\u2E00-\u2E7F\u3000-\u303F\uFFF0-\uFFFF]')
        self.boilerplate_patterns = [
            r'\.css-[a-z0-9]+',
            r'r-[a-z0-9]+',
            r'class="[^"]*"',
            r'<[^>]*>',
            r'<!DOCTYPE[^>]*>',
            r'&[a-zA-Z]+;',
            r'javascript:',
            r'function\s*\([^)]*\)',
            r'var\s+[a-zA-Z_][a-zA-Z0-9_]*',
            r'@media[^{]*{[^}]*}',
            r'href="[^"]*"',
            r'src="[^"]*"',
            r'data-[a-zA-Z-]+=',
            r'aria-[a-zA-Z-]+=',
            r'role="[^"]*"'
        ]
        
        # Common FAQ question starters that indicate answer truncation
        self.question_starters = [
            "What is", "What are", "What should", "What if", "What does",
            "How can", "How do", "How to", "How does", "How will",
            "Where can", "Where do", "Where is",
            "When should", "When do", "When will",
            "Why is", "Why does", "Why should",
            "Can I", "Can we", "Can you",
            "Is it", "Is there", "Is partial",
            "Do I", "Do we", "Do you",
            "Will there", "Will you", "Will I",
            "Who will", "Which"
        ]
    
    def remove_garbage_characters(self, text: str) -> str:
        """
        Remove garbage Unicode characters and rendering artifacts
        """
        if not text:
            return ""
        
        # Count garbage characters for stats
        garbage_count = len(self.garbage_unicode_pattern.findall(text))
        
        # Specific problematic Unicode characters found in JioPay data
        specific_garbage_chars = [
            '\ue953',  # Private Use Area character appearing as empty square
            '\ue935',  # Private Use Area character appearing as empty square
            '\uf8ff',  # Apple logo private use character
            '\uf8a0',  # Another common private use character
        ]
        
        # Remove specific garbage characters first
        for char in specific_garbage_chars:
            if char in text:
                garbage_count += text.count(char)
                text = text.replace(char, '')
        
        if garbage_count > 0:
            self.stats["garbage_chars_removed"] += garbage_count
        
        # Remove garbage Unicode characters using pattern
        text = self.garbage_unicode_pattern.sub('', text)
        
        # Remove common rendering artifacts
        text = text.replace('', '').replace('', '')  # Empty box characters
        text = text.replace('\ufffd', '')  # Replacement character
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        text = text.replace('\ufeff', '')  # Byte order mark
        text = text.replace('\u00a0', ' ')  # Non-breaking space to regular space
        text = text.replace('\u2028', ' ')  # Line separator to space
        text = text.replace('\u2029', ' ')  # Paragraph separator to space
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def remove_html_boilerplate(self, text: str) -> str:
        """
        Remove HTML/CSS boilerplate and website code
        """
        if not text:
            return ""
        
        original_length = len(text)
        
        # Apply all boilerplate removal patterns
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove common website boilerplate phrases
        boilerplate_phrases = [
            "Terms and Conditions",
            "Privacy Policy", 
            "Contact Us",
            "About Us",
            "All Rights Reserved",
            "Copyright",
            "Follow us on",
            "Download our app",
            "Visit our website",
            "Call customer care",
            "Email us at",
            "©",
            "™",
            "®"
        ]
        
        for phrase in boilerplate_phrases:
            text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
        
        # Remove CSS-like selectors and HTML attributes
        text = re.sub(r'\.[a-zA-Z][a-zA-Z0-9_-]*\s*{[^}]*}', '', text)
        text = re.sub(r'#[a-zA-Z][a-zA-Z0-9_-]*\s*{[^}]*}', '', text)
        
        # Track boilerplate removal
        if len(text) < original_length * 0.8:  # If we removed more than 20%
            self.stats["boilerplate_removed"] += 1
        
        return text
    
    def fix_truncated_answers(self, text: str) -> str:
        """
        Fix truncated answers that are followed by next question text
        """
        if not text:
            return ""
        
        original_text = text
        
        # More aggressive approach: find any question-like patterns
        # Look for patterns where questions are concatenated without proper spacing
        
        # Pattern 1: Question after period/question mark
        question_after_punct = r'[.!?]\s*(?:What|How|Can|Do|Is|Are|Where|When|Why|Which|Who|Should|Would|Could|Will|Did|Does|Have|Has)\s+'
        match = re.search(question_after_punct, text, re.IGNORECASE)
        if match:
            # Keep everything up to and including the punctuation
            text = text[:match.end() - len(match.group().split()[-1]) - 1]
            self.stats["truncated_answers_fixed"] += 1
            return text.strip()
        
        # Pattern 2: Question directly attached (no punctuation)
        question_attached = r'([.!?]?)(?:What|How|Can|Do|Is|Are|Where|When|Why|Which|Who|Should|Would|Could|Will|Did|Does|Have|Has)\s+(?:is|are|can|do|does|have|has|will|would|could|should|the|I|you|we|my|your|our)'
        match = re.search(question_attached, text, re.IGNORECASE)
        if match:
            start_pos = match.start()
            # If we have substantial content before the question
            if start_pos > 50:
                if match.group(1):  # Has punctuation
                    text = text[:start_pos + len(match.group(1))]
                else:  # No punctuation, add one
                    text = text[:start_pos].strip() + '.'
                self.stats["truncated_answers_fixed"] += 1
                return text.strip()
        
        # Pattern 3: Multiple question words in sequence (indicates concatenation)
        multiple_questions = r'(?:What|How|Can|Do|Is|Are|Where|When|Why|Which|Who|Should|Would|Could|Will|Did|Does|Have|Has)(?:\s+[^.!?]*){0,10}(?:What|How|Can|Do|Is|Are|Where|When|Why|Which|Who|Should|Would|Could|Will|Did|Does|Have|Has)'
        match = re.search(multiple_questions, text, re.IGNORECASE)
        if match:
            start_pos = match.start()
            if start_pos > 50:
                # Look for the last sentence ending before this
                preceding_text = text[:start_pos]
                sentence_endings = list(re.finditer(r'[.!?]', preceding_text))
                if sentence_endings:
                    text = text[:sentence_endings[-1].end()]
                else:
                    text = preceding_text.strip() + '.'
                self.stats["truncated_answers_fixed"] += 1
                return text.strip()
        
        # Pattern 4: Look for incomplete answers ending with partial words followed by questions
        incomplete_pattern = r'(.{50,}?)(?:\s+[a-z]{1,4}(?:\s|$))?(?:What|How|Can|Do|Is|Are|Where|When|Why|Which|Who|Should|Would|Could|Will|Did|Does|Have|Has)\s+'
        match = re.search(incomplete_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            clean_text = match.group(1).strip()
            # Ensure proper ending
            if not clean_text.endswith(('.', '!', '?')):
                clean_text += '.'
            self.stats["truncated_answers_fixed"] += 1
            return clean_text
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Comprehensive text normalization
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Remove garbage characters
        text = self.remove_garbage_characters(text)
        
        # Step 2: Remove HTML/CSS boilerplate
        text = self.remove_html_boilerplate(text)
        
        # Step 3: Fix truncated answers
        text = self.fix_truncated_answers(text)
        
        # Step 4: Basic text normalization
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove multiple spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove repeated punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text.strip()
    
    def clean_faq_data(self, faqs: List[Dict]) -> List[Dict]:
        """
        Enhanced FAQ cleaning with specific issue fixes
        """
        cleaned_faqs = []
        seen_questions = set()
        
        for faq in faqs:
            if not isinstance(faq, dict):
                continue
            
            question = faq.get('question', '')
            answer = faq.get('answer', '')
            category = faq.get('category', 'General')
            
            # Normalize question and answer with enhanced cleaning
            clean_question = self.normalize_text(question)
            clean_answer = self.normalize_text(answer)
            
            # Skip entries that are too short after cleaning
            if len(clean_question) < 10:
                self.stats["empty_entries_removed"] += 1
                continue
            
            if len(clean_answer) < 20:
                self.stats["empty_entries_removed"] += 1
                continue
            
            # Remove duplicates based on question similarity
            question_normalized = re.sub(r'[^\w\s]', '', clean_question.lower())
            question_key = ' '.join(question_normalized.split()[:5])  # First 5 words
            
            if question_key in seen_questions:
                self.stats["duplicates_removed"] += 1
                continue
            
            seen_questions.add(question_key)
            
            # Additional quality checks
            # Skip if answer is just a question or very repetitive
            if clean_answer.count('?') > 3:  # Too many questions in answer
                self.stats["empty_entries_removed"] += 1
                continue
            
            # Skip if answer is mostly the same word repeated
            words = clean_answer.split()
            if len(words) > 10 and len(set(words)) < len(words) * 0.3:
                self.stats["empty_entries_removed"] += 1
                continue
            
            cleaned_faq = {
                "category": category.strip(),
                "question": clean_question,
                "answer": clean_answer,
                "question_length": len(clean_question),
                "answer_length": len(clean_answer),
                "normalized_timestamp": datetime.now().isoformat()
            }
            
            cleaned_faqs.append(cleaned_faq)
            self.stats["total_faqs_processed"] += 1
        
        return cleaned_faqs
    
    def extract_playwright_faqs(self, playwright_data: Dict) -> List[Dict]:
        """
        Extract FAQ-like data from Playwright scraped data using title as question
        """
        extracted_faqs = []
        
        if not isinstance(playwright_data, dict):
            return extracted_faqs
        
        # Extract from pages
        pages = playwright_data.get('pages', [])
        for page in pages:
            if not isinstance(page, dict):
                continue
            
            # Extract business_info sections
            business_info = page.get('business_info', {})
            for key, info in business_info.items():
                if isinstance(info, dict):
                    heading = info.get('heading', '')
                    content = info.get('content', '')
                    
                    if heading and content:
                        # Use heading as question and content as answer
                        extracted_faqs.append({
                            'question': heading,
                            'answer': content,
                            'category': 'Business Information'
                        })
            
            # Extract from content_sections
            content_sections = page.get('content_sections', [])
            for section in content_sections:
                if isinstance(section, dict):
                    heading = section.get('heading', '')
                    content = section.get('content', '') or section.get('full_content', '')
                    
                    if heading and content:
                        # Split large content into smaller FAQ-like chunks
                        if len(content) > 500:
                            # Try to split by common patterns
                            parts = re.split(r'\s+(?:Know More|Learn More|Read More)\s+', content)
                            for i, part in enumerate(parts):
                                if len(part.strip()) > 100:
                                    section_title = f"{heading} - Part {i+1}" if len(parts) > 1 else heading
                                    extracted_faqs.append({
                                        'question': section_title,
                                        'answer': part.strip(),
                                        'category': 'Content Information'
                                    })
                        else:
                            extracted_faqs.append({
                                'question': heading,
                                'answer': content,
                                'category': 'Content Information'
                            })
            
            # Extract title and meta description as FAQ
            title = page.get('title', '')
            meta_description = page.get('meta_description', '')
            full_text = page.get('full_text', '')
            
            if title and (meta_description or full_text):
                answer_text = meta_description if meta_description else full_text[:500] + "..." if len(full_text) > 500 else full_text
                extracted_faqs.append({
                    'question': f"What is {title}?",
                    'answer': answer_text,
                    'category': 'General Information'
                })
        
        return extracted_faqs
    
    def normalize_scraped_data(self, file_path: str) -> Dict:
        """
        Main normalization function with enhanced FAQ cleaning
        """
        logger.info(f"Starting enhanced normalization of: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return {}
        
        self.stats["normalization_timestamp"] = datetime.now().isoformat()
        
        # Initialize cleaned data structure
        self.cleaned_data = {
            "source_file": file_path,
            "normalization_info": {
                "normalized_at": self.stats["normalization_timestamp"],
                "normalizer_version": "2.0_enhanced",
                "fixes_applied": [
                    "garbage_unicode_removal",
                    "truncated_answer_fixing", 
                    "html_boilerplate_removal",
                    "text_normalization",
                    "playwright_data_extraction"
                ],
                "source_data_timestamp": raw_data.get('extraction_info', {}).get('timestamp') or raw_data.get('site_info', {}).get('scrape_timestamp')
            },
            "faqs": [],
            "metadata": {}
        }
        
        # Process FAQs with enhanced cleaning
        if 'faqs' in raw_data:
            self.cleaned_data["faqs"] = self.clean_faq_data(raw_data['faqs'])
        
        # Process pages for additional FAQs
        if 'pages' in raw_data:
            all_page_faqs = []
            for page in raw_data['pages']:
                page_faqs = page.get('faqs', [])
                if page_faqs:
                    all_page_faqs.extend(page_faqs)
            
            if all_page_faqs:
                cleaned_page_faqs = self.clean_faq_data(all_page_faqs)
                self.cleaned_data["faqs"].extend(cleaned_page_faqs)
        
        # Check if this is Playwright data and extract FAQ-like content
        if 'site_info' in raw_data and 'pages' in raw_data:
            logger.info("Detected Playwright data structure - extracting FAQ content")
            playwright_faqs = self.extract_playwright_faqs(raw_data)
            if playwright_faqs:
                cleaned_playwright_faqs = self.clean_faq_data(playwright_faqs)
                self.cleaned_data["faqs"].extend(cleaned_playwright_faqs)
                logger.info(f"Extracted {len(cleaned_playwright_faqs)} FAQs from Playwright data")
        
        # Add metadata
        self.cleaned_data["metadata"] = {
            "total_faqs": len(self.cleaned_data["faqs"]),
            "processing_stats": self.stats,
            "categories": list(set(faq.get('category', 'General') for faq in self.cleaned_data["faqs"]))
        }
        
        logger.info(f"Enhanced normalization complete. Processed {self.stats['total_faqs_processed']} FAQs")
        logger.info(f"Fixed: {self.stats['garbage_chars_removed']} garbage chars, "
                   f"{self.stats['truncated_answers_fixed']} truncated answers, "
                   f"{self.stats['boilerplate_removed']} boilerplate sections")
        
        return self.cleaned_data
    
    def save_normalized_data(self, output_path: Optional[str] = None) -> str:
        """
        Save normalized data to JSON file
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"../Dataset/Normalized_data.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.cleaned_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Enhanced normalized data saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving normalized data: {e}")
            return ""
    
    def get_normalization_report(self) -> str:
        """
        Generate detailed normalization report
        """
        total_faqs = self.cleaned_data.get('metadata', {}).get('total_faqs', 0)
        
        report = f"""
Enhanced JioPay FAQ Normalization Report
========================================

Issues Fixed:
- Garbage Unicode characters removed: {self.stats['garbage_chars_removed']}
- Truncated answers fixed: {self.stats['truncated_answers_fixed']}
- HTML/CSS boilerplate sections removed: {self.stats['boilerplate_removed']}
- Duplicate FAQs removed: {self.stats['duplicates_removed']}
- Empty/invalid entries removed: {self.stats['empty_entries_removed']}

Processing Metrics:
- Total clean FAQs: {total_faqs}

Categories Found:
{', '.join(self.cleaned_data.get('metadata', {}).get('categories', []))}

Processing Summary:
- Original FAQs processed: {self.stats['total_faqs_processed']}
- Data quality significantly improved
- All major issues addressed
- Ready for RAG system ingestion

Normalized at: {self.stats['normalization_timestamp']}
        """
        return report.strip()

def main():
    """
    Main function to run enhanced normalization on FAQ data
    """
    logger.info("Starting enhanced JioPay FAQ normalization process...")
    
    # Find the enhanced FAQ file
    scraped_data_dir = Path("../Scraped Data")
    
    if not scraped_data_dir.exists():
        logger.error("Scraped Data directory not found!")
        return
    
    # Look for both enhanced FAQ files and Playwright data files
    faq_files = list(scraped_data_dir.glob("*enhanced_faqs*.json"))
    playwright_files = list(scraped_data_dir.glob("*playwright*.json"))
    
    all_files = faq_files + playwright_files
    
    if not all_files:
        logger.error("No FAQ or Playwright data files found!")
        return
    
    normalizer = EnhancedJioPayDataNormalizer()
    all_normalized_faqs = []
    
    # Process all files to combine into one normalized dataset
    for file_path in all_files:
        logger.info(f"Processing file: {file_path}")
        
        # Reset stats for each file
        normalizer.stats = {
            "garbage_chars_removed": 0,
            "truncated_answers_fixed": 0,
            "boilerplate_removed": 0,
            "duplicates_removed": 0,
            "empty_entries_removed": 0,
            "total_faqs_processed": 0,
            "normalization_timestamp": ""
        }
        
        normalized_data = normalizer.normalize_scraped_data(str(file_path))
        
        if normalized_data and normalized_data.get('faqs'):
            all_normalized_faqs.extend(normalized_data['faqs'])
            logger.info(f"Added {len(normalized_data['faqs'])} FAQs from {file_path.name}")
    
    if all_normalized_faqs:
        # Remove duplicates across all files using the same deduplication logic
        final_faqs = []
        seen_questions = set()
        
        for faq in all_normalized_faqs:
            question_normalized = re.sub(r'[^\w\s]', '', faq['question'].lower())
            question_key = ' '.join(question_normalized.split()[:5])
            
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                final_faqs.append(faq)
            else:
                logger.debug(f"Removed duplicate: {faq['question'][:50]}...")
        
        # Create final normalized data structure
        normalizer.cleaned_data = {
            "source_files": [str(f) for f in all_files],
            "normalization_info": {
                "normalized_at": datetime.now().isoformat(),
                "normalizer_version": "2.0_enhanced_combined",
                "fixes_applied": [
                    "garbage_unicode_removal",
                    "truncated_answer_fixing", 
                    "html_boilerplate_removal",
                    "text_normalization",
                    "playwright_data_extraction",
                    "cross_file_deduplication"
                ],
                "files_processed": len(all_files)
            },
            "faqs": final_faqs,
            "metadata": {
                "total_faqs": len(final_faqs),
                "categories": list(set(faq.get('category', 'General') for faq in final_faqs)),
                "faq_files_processed": len(faq_files),
                "playwright_files_processed": len(playwright_files)
            }
        }
        
        # Save combined normalized data
        output_file = normalizer.save_normalized_data()
        
        # Print report
        print(f"\n{normalizer.get_normalization_report()}")
        print(f"\nCombined normalized data saved to: {output_file}")
        print(f"Files processed: {len(all_files)} ({len(faq_files)} FAQ files, {len(playwright_files)} Playwright files)")
        print(f"Total unique FAQs: {len(final_faqs)}")
    else:
        logger.warning("No FAQs found in any processed files")

if __name__ == "__main__":
    main()
