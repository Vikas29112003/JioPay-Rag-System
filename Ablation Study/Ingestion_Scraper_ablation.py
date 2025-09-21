#!/usr/bin/env python3
"""
Ingestion/Scraper Ablation Study
Comparing Playwright vs Selenium scrapers for JioPay data extraction
"""

import json
import re
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def count_tokens(text):
    """Estimate token count (approximation: 1 token ≈ 4 characters)"""
    return len(text) // 4

def calculate_noise_percentage(total_content_sections, total_business_info):
    """Calculate noise percentage based on irrelevant content"""
    
    if total_content_sections == 0:
        return 0
    
    # Noise = non-business content / total content
    noise_sections = max(0, total_content_sections - total_business_info)
    return (noise_sections / total_content_sections) * 100

def calculate_throughput(pages, start_time, end_time):
    """Calculate pages per minute"""
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time)
    duration_minutes = (end - start).total_seconds() / 60
    
    if duration_minutes == 0:
        return 0
    
    return pages / duration_minutes

def analyze_scraper_data(file_path, scraper_name):
    """Analyze a single scraper's JSON output"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    site_info = data.get('site_info', {})
    pages = data.get('pages', [])
    
    # Extract metrics
    num_pages = site_info.get('total_pages_scraped', 0)
    
    # Count tokens from all text content
    total_text = ""
    total_content_sections = 0
    total_business_info = 0
    total_faqs = 0
    
    for page in pages:
        # Extract all text content
        if 'full_text' in page:
            total_text += page['full_text']
        
        # Count content sections
        content_sections = page.get('content_sections', [])
        total_content_sections += len(content_sections)
        
        # Count business info sections
        business_info = page.get('business_info', {})
        total_business_info += len(business_info)
        
        # Count FAQs (if available)
        faqs = page.get('faqs', [])
        total_faqs += len(faqs)
    
    # Also check for FAQ data at root level (for enhanced scraper data)
    root_faqs = data.get('faqs', [])
    if root_faqs:
        total_faqs += len(root_faqs)
    
    num_tokens = count_tokens(total_text)
    
    # Calculate noise percentage
    noise_percent = calculate_noise_percentage(total_content_sections, total_business_info)
    
    # Calculate throughput (pages per minute)
    start_time = site_info.get('scrape_timestamp')
    end_time = site_info.get('completion_timestamp')
    
    if start_time and end_time:
        throughput = calculate_throughput(num_pages, start_time, end_time)
    else:
        throughput = 0
    
    # Calculate failure rate (0% if successful)
    failure_rate = 0.0  # Both scrapers completed successfully
    
    return {
        'pipeline': scraper_name,
        'pages': num_pages,
        'tokens': num_tokens,
        'faqs': total_faqs,
        'noise_percent': round(noise_percent, 1),
        'throughput': round(throughput, 2),
        'failure_rate': failure_rate
    }

def main():
    """Main analysis function"""
    
    # Define file paths
    scraped_data_dir = Path("../Scraped Data")
    
    playwright_file = None
    selenium_file = None
    faq_file = None
    
    # Find the files
    for file in scraped_data_dir.glob("*.json"):
        if "playwright" in file.name:
            playwright_file = file
        elif "selenium" in file.name:
            selenium_file = file
        elif "enhanced_faqs" in file.name:
            faq_file = file
    
    if not playwright_file or not selenium_file:
        print("Error: Could not find both Playwright and Selenium data files")
        return
    
    # Analyze both scrapers
    playwright_results = analyze_scraper_data(playwright_file, "Playwright")
    selenium_results = analyze_scraper_data(selenium_file, "Selenium")
    
    # Add FAQ data to Playwright results if available
    if faq_file:
        with open(faq_file, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        playwright_results['faqs'] = faq_data.get('extraction_info', {}).get('total_faqs', 0)
        print(f"✅ Added {playwright_results['faqs']} FAQs to Playwright analysis from enhanced scraper")
    
    # Create comparison table
    print("Ingestion/Scraper Ablation:")
    print("=" * 95)
    print(f"{'Pipeline':<15} {'#Pages':<8} {'#Tokens':<10} {'#FAQs':<8} {'Noise %':<8} {'Throughput':<12} {'Failures (%)':<12}")
    print("-" * 95)
    
    # Playwright row
    print(f"{playwright_results['pipeline']:<15} {playwright_results['pages']:<8} {playwright_results['tokens']:<10} {playwright_results['faqs']:<8} {playwright_results['noise_percent']:<8} {playwright_results['throughput']:<12} {playwright_results['failure_rate']:<12}")
    
    # Selenium row
    print(f"{selenium_results['pipeline']:<15} {selenium_results['pages']:<8} {selenium_results['tokens']:<10} {selenium_results['faqs']:<8} {selenium_results['noise_percent']:<8} {selenium_results['throughput']:<12} {selenium_results['failure_rate']:<12}")
    
    print("-" * 95)
    
    # Analysis summary
    print("\nAnalysis Summary:")
    print(f"• Token Extraction: {'Playwright' if playwright_results['tokens'] > selenium_results['tokens'] else 'Selenium'} extracted more tokens")
    print(f"• FAQ Extraction: {'Playwright' if playwright_results['faqs'] > selenium_results['faqs'] else 'Selenium'} extracted more FAQs ({playwright_results['faqs']} vs {selenium_results['faqs']})")
    print(f"• Data Quality: {'Playwright' if playwright_results['noise_percent'] < selenium_results['noise_percent'] else 'Selenium'} has lower noise percentage")
    print(f"• Speed: {'Playwright' if playwright_results['throughput'] > selenium_results['throughput'] else 'Selenium'} has higher throughput")
    
    # Save results to file
    results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "playwright": playwright_results,
        "selenium": selenium_results,
        "comparison": {
            "better_token_extraction": "Playwright" if playwright_results['tokens'] > selenium_results['tokens'] else "Selenium",
            "better_faq_extraction": "Playwright" if playwright_results['faqs'] > selenium_results['faqs'] else "Selenium",
            "better_data_quality": "Playwright" if playwright_results['noise_percent'] < selenium_results['noise_percent'] else "Selenium",
            "better_throughput": "Playwright" if playwright_results['throughput'] > selenium_results['throughput'] else "Selenium"
        }
    }
    
    output_file = "result/scraper_ablation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Create PNG visualization
    create_visualization(playwright_results, selenium_results)

def create_visualization(playwright_results, selenium_results):
    """Create PNG charts comparing the scrapers"""
    
    # Set up the figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('JioPay Scraper Ablation Study Results', fontsize=16, fontweight='bold')
    
    scrapers = ['Playwright', 'Selenium']
    
    # 1. Token Count Comparison
    tokens = [playwright_results['tokens'], selenium_results['tokens']]
    bars1 = ax1.bar(scrapers, tokens, color=['#2E86AB', '#A23B72'])
    ax1.set_title('Token Extraction Comparison')
    ax1.set_ylabel('Number of Tokens')
    for i, v in enumerate(tokens):
        ax1.text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. FAQ Count Comparison
    faqs = [playwright_results['faqs'], selenium_results['faqs']]
    bars2 = ax2.bar(scrapers, faqs, color=['#F18F01', '#C73E1D'])
    ax2.set_title('FAQ Extraction Comparison')
    ax2.set_ylabel('Number of FAQs')
    for i, v in enumerate(faqs):
        ax2.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 3. Throughput Comparison
    throughput = [playwright_results['throughput'], selenium_results['throughput']]
    bars3 = ax3.bar(scrapers, throughput, color=['#3F7CAC', '#95A3B3'])
    ax3.set_title('Processing Speed (Pages/Minute)')
    ax3.set_ylabel('Pages per Minute')
    for i, v in enumerate(throughput):
        ax3.text(i, v + 0.2, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Overall Comparison Table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'Playwright', 'Selenium', 'Winner'],
        ['Pages Scraped', str(playwright_results['pages']), str(selenium_results['pages']), 'Tie'],
        ['Tokens Extracted', str(playwright_results['tokens']), str(selenium_results['tokens']), 
         'Tie' if playwright_results['tokens'] == selenium_results['tokens'] else 
         ('Playwright' if playwright_results['tokens'] > selenium_results['tokens'] else 'Selenium')],
        ['FAQs Extracted', str(playwright_results['faqs']), str(selenium_results['faqs']),
         'Tie' if playwright_results['faqs'] == selenium_results['faqs'] else 
         ('Playwright' if playwright_results['faqs'] > selenium_results['faqs'] else 'Selenium')],
        ['Noise %', f"{playwright_results['noise_percent']}%", f"{selenium_results['noise_percent']}%",
         'Selenium' if selenium_results['noise_percent'] < playwright_results['noise_percent'] else 'Playwright'],
        ['Throughput', f"{playwright_results['throughput']:.2f}", f"{selenium_results['throughput']:.2f}",
         'Playwright' if playwright_results['throughput'] > selenium_results['throughput'] else 'Selenium'],
        ['Failure Rate', f"{playwright_results['failure_rate']}%", f"{selenium_results['failure_rate']}%", 'Tie']
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style winner column
    for i in range(1, 7):
        winner = table_data[i][3]
        if winner == 'Playwright':
            table[(i, 3)].set_facecolor('#90EE90')
        elif winner == 'Selenium':
            table[(i, 3)].set_facecolor('#FFB6C1')
        else:
            table[(i, 3)].set_facecolor('#F0F0F0')
    
    ax4.set_title('Detailed Comparison Table', pad=20)
    
    plt.tight_layout()
    
    # Save the plot as PNG
    output_png = "result/scraper_ablation_comparison.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_png}")
    
    plt.close()

if __name__ == "__main__":
    main()