import os
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from tqdm import tqdm

# Initialize Result Storage
results = []

# Resume Keywords (Expanded with domain-specific terms)
Area_with_key_term = {
    'Data science': ['algorithm', 'analytics', 'hadoop', 'machine learning', 'data mining', 'python',
                     'statistics', 'data', 'statistical analysis', 'data wrangling', 'algebra', 'probability',
                     'visualization', 'big data', 'predictive modeling', 'artificial intelligence'],
    'Programming': ['python', 'r programming', 'sql', 'c++', 'scala', 'julia', 'tableau',
                    'javascript', 'powerbi', 'code', 'coding', 'html', 'css', 'java'],
    'Experience': ['project', 'years', 'company', 'excellency', 'promotion', 'award',
                   'outsourcing', 'work in progress', 'team lead', 'cross-functional'],
    'Management skill': ['administration', 'budget', 'cost', 'direction', 'feasibility analysis',
                         'finance', 'leader', 'leadership', 'management', 'milestones', 'planning',
                         'problem', 'project', 'risk', 'schedule', 'stakeholders', 'english'],
    'Data analytics': ['api', 'big data', 'clustering', 'code', 'coding', 'data', 'database',
                       'data mining', 'data science', 'deep learning', 'hadoop',
                       'hypothesis test', 'machine learning', 'dbms', 'modeling', 'nlp',
                       'predictive', 'text mining', 'visualization'],
    'Statistics': ['parameter', 'variable', 'ordinal', 'ratio', 'nominal', 'interval', 'descriptive',
                   'inferential', 'linear', 'correlations', 'probability',
                   'regression', 'mean', 'variance', 'standard deviation'],
    'Machine learning': ['supervised learning', 'unsupervised learning', 'ann', 'artificial neural network',
                         'overfitting', 'computer vision', 'natural language processing',
                         'database', 'tensorflow', 'pytorch', 'scikit-learn'],
    'Data analyst': ['data collection', 'data cleaning', 'data processing', 'interpreting data',
                     'streamlining data', 'visualizing data', 'statistics',
                     'tableau', 'tables', 'analytical'],
    'Software': ['django', 'cloud', 'gcp', 'aws', 'javascript', 'react', 'redux',
                 'es6', 'node.js', 'typescript', 'html', 'css', 'ui', 'ci/cd', 'cashflow','C','JAVA'],
    'Web skill': ['web design', 'branding', 'graphic design', 'seo', 'marketing', 'logo design', 'video editing',
                  'es6', 'node.js', 'typescript', 'html', 'css', 'ci/cd'],
    'Personal Skill': ['leadership', 'team work', 'integrity', 'public speaking', 'team leadership', 'problem solving',
                       'loyalty', 'quality', 'performance improvement', 'six sigma', 'quality circles', 'quality tools',
                       'process improvement', 'capability analysis', 'control'],
    'Accounting': ['communication', 'sales', 'sales process', 'solution selling', 'crm',
                   'sales management', 'sales operations', 'marketing', 'direct sales', 'trends', 'b2b', 'marketing strategy', 'saas',
                   'business development'],
    'Sales & marketing': ['retail', 'manufacture', 'corporate', 'goodssale', 'consumer',
                          'package', 'fmcg', 'account', 'management', 'lead generation', 'cold calling', 'customer service',
                          'inside sales', 'sales', 'promotion'],
    'Graphic': ['brand identity', 'editorial design', 'design', 'branding', 'logo design',
                'letterhead design', 'business card design', 'brand strategy', 'stationery design', 'graphic design',
                'exhibition graphic design'],
    'Content skill': ['editing', 'creativity', 'content idea', 'problem solving', 'writer',
                      'content thinker', 'copy editor', 'researchers', 'technology geek', 'public speaking', 'online marketing'],
    'Graphical content': ['photographer', 'videographer', 'graphic artist', 'copywriter', 'search engine optimization',
                          'seo', 'social media', 'page insight', 'gain audience'],
    'Finance': ['financial reporting', 'budgeting', 'forecasting', 'strong analytical thinking', 'financial planning',
                 'payroll tax', 'accounting', 'productivity', 'reporting costs', 'balance sheet',
                 'financial statements'],
    'Health/Medical': ['abdominal surgery', 'laparoscopy', 'trauma surgery', 'adult intensive care',
                       'pain management', 'cardiology', 'patient', 'surgery', 'hospital', 'healthcare', 'doctor', 'medicine'],
    'Language': ['english', 'malay', 'mandarin', 'bangla', 'hindi', 'tamil', 'spanish', 'french', 'urdu', 'chinese']
}

def extract_text_from_pdf(pdf_path):
    """Extracts text content from PDF file using PyPDF2 library"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def clean_text(text):
    """Cleans and normalizes text by removing punctuation, numbers and converting to lowercase"""
    text = text.lower()
    text = re.sub(r'[0-9]+', '', text)
    return text.translate(str.maketrans('', '', string.punctuation))

def score_resume(text):
    """Calculates scores for each domain based on keyword matches"""
    score_dict = {domain: 0 for domain in Area_with_key_term}
    for domain, keywords in Area_with_key_term.items():
        for word in keywords:
            if word in text:
                score_dict[domain] += 1
    return score_dict

def suggest_role(scores):
    """Determines job role suggestion based on score thresholds"""
    total = sum(scores.values())
    ds = scores['Data science']
    prog = scores['Programming']
    exp = scores['Experience']
    mgmt = scores['Management skill']
    da = scores['Data analytics']
    stat = scores['Statistics']
    ml = scores['Machine learning']
    analyst = scores['Data analyst']
    sw = scores['Software']
    web = scores['Web skill']
    ps = scores['Personal Skill']
    acc = scores['Accounting']
    sm = scores['Sales & marketing']
    graphic = scores['Graphic']
    content = scores['Content skill']
    g_content = scores['Graphical content']
    fin = scores['Finance']
    med = scores['Health/Medical']
    lang = scores['Language']
    
    role_conditions = [
        (total >= 50 and ps >= 2 and lang >= 1 and stat >= 9, "Junior Data Scientist"),
        (total >= 40 and ps >= 2 and lang >= 1 and ds >= 10, "Junior Data Scientist"),
        (total >= 60 and lang >= 1 and da >= 8, "Junior Data Scientist"),
        (total >= 30 and prog >= 3 and da >= 5, "Data Analyst"),
        (total >= 20 and exp >= 2 and sw >= 10, "Software Engineer"),
        (total >= 18 and graphic >= 5 and web >= 10, "Web Developer"),
        (total >= 50 and acc >= 10, "Account Executive"),
        (total >= 20 and sm >= 10, "Sales Representative"),
        (total >= 25 and content >= 8, "Content Creator"),
        (total >= 30 and fin >= 10, "Senior Accountant"),
        (total >= 20 and med >= 10, "Medical Professional")
    ]

    # Check conditions in order and return first match
    for condition, role in role_conditions:
        if condition:
            return role
            
    return "Consider for other roles"

def generate_chart(file, scores, output_dir):
    """Generates and saves a pie chart visualization"""
    df_score = pd.DataFrame(list(scores.items()), columns=['Domain', 'Score'])
    
    # Filter out zero scores and sort
    df_score = df_score[df_score['Score'] > 0].sort_values('Score', ascending=False)
    
    if df_score.empty:
        print(f"No score data to visualize for {file}")
        return
    
    try:
        plt.figure(figsize=(10, 8))
        plt.pie(df_score['Score'], 
                labels=df_score['Domain'],
                autopct='%1.1f%%',
                startangle=140,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        
        plt.title(f"Skill Distribution\n{file[:-4]}", pad=20)
        plt.axis('equal')
        
        chart_path = os.path.join(output_dir, 'graphs', f"{file.replace('.pdf', '')}_chart.png")
        plt.savefig(chart_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Chart saved: {chart_path}")
    except Exception as e:
        print(f"Error generating chart for {file}: {str(e)}")

def generate_summary_chart(summary_df):
    """Generates a summary bar chart for all resumes processed"""
    plt.figure(figsize=(12, 6))
    plt.barh(summary_df['Filename'], summary_df['Total Score'], color='darkblue')
    plt.xlabel('Total Score')
    plt.title('Summary of All Resumes Processed')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('output/summary_chart.png', dpi=200)
    plt.close()
    print("Summary chart saved: output/summary_chart.png")

def main(resume_dir='./Resume'):
    """Main processing function that handles resume analysis"""
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'graphs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)

    csv_rows = []
    
    pdf_files = [f for f in os.listdir(resume_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in directory")
        return

    for file in tqdm(pdf_files, desc="Processing Resumes"):
        try:
            filepath = os.path.join(resume_dir, file)
            text = extract_text_from_pdf(filepath)
            cleaned_text = clean_text(text)
            scores = score_resume(cleaned_text)
            role = suggest_role(scores)
            
            # Prepare row for CSV output
            row = {'Filename': file, 'Total Score': sum(scores.values()), 'Suggestion': role}
            row.update(scores)
            csv_rows.append(row)
            
            # Console output for each resume
            print(f"\n============================== Analyzing: {file} ==============================")
            print("\nScore Breakdown:")
            for key, value in scores.items():
                print(f"{key}: {value}")

                
            print(f"Total Score: {sum(scores.values())}")
            print(f"Recommendation: {role}")


            # Generate visualization
            generate_chart(file, scores, output_dir)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    # Save results to CSV
    if csv_rows:
        results_df = pd.DataFrame(csv_rows)
        results_df.to_csv(os.path.join(output_dir, 'results', 'resume_analysis_results.csv'), index=False)
        print("\nAnalysis complete. Results saved to output/results/resume_analysis_results.csv")
        
        # Generate summary chart
        generate_summary_chart(results_df)
    else:
        print("\nNo valid results to save")

if __name__ == "__main__":
    main()
