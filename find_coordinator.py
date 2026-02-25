import pandas as pd
import requests
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import zipfile

ZIP_URL = "https://cordis.europa.eu/data/cordis-HORIZONprojects-csv.zip"

def download_and_extract(url, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    project_csv = os.path.join(data_dir, 'project.csv')
    org_csv = os.path.join(data_dir, 'organization.csv')
    
    if os.path.exists(project_csv) and os.path.exists(org_csv):
        print(f"Using cached data in {data_dir}.")
        return

    zip_path = os.path.join(data_dir, "cordis.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {url}...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading: {e}")
            raise e
            
    print("Extracting ZIP archive...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")

def load_data():
    data_dir = 'cordis_data'
    download_and_extract(ZIP_URL, data_dir)

    print("Loading datasets...")
    try:
        # CORDIS CSVs are usually semicolon separated. Let's be lenient with parse errors if any.
        projects_df = pd.read_csv(os.path.join(data_dir, 'project.csv'), sep=';', on_bad_lines='skip', low_memory=False)
        orgs_df = pd.read_csv(os.path.join(data_dir, 'organization.csv'), sep=';', on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None
    print(f"Loaded {len(projects_df)} projects and {len(orgs_df)} organizations.")
    return projects_df, orgs_df

def find_coordinators(query, n_top=5):
    projects_df, orgs_df = load_data()
    if projects_df is None or orgs_df is None:
        return

    # Check for expected columns
    if 'title' not in projects_df.columns:
        print(f"Available project columns: {list(projects_df.columns)}")
        # Handle possible different separator
        if len(projects_df.columns) == 1:
            print("It seems the separator might be a comma instead of a semicolon. Trying comma...")
            projects_df = pd.read_csv('cordis-HEprojects.csv', sep=',', low_memory=False, on_bad_lines='skip')
            orgs_df = pd.read_csv('cordis-HEorganizations.csv', sep=',', low_memory=False, on_bad_lines='skip')

    # Combine text fields for projects
    print("Preparing project data...")
    title_col = 'title' if 'title' in projects_df.columns else ''
    obj_col = 'objective' if 'objective' in projects_df.columns else ''
    topics_col = 'topics' if 'topics' in projects_df.columns else ''
    
    text_data = []
    if title_col: text_data.append(projects_df[title_col].fillna('').astype(str))
    if obj_col: text_data.append(projects_df[obj_col].fillna('').astype(str))
    if topics_col: text_data.append(projects_df[topics_col].fillna('').astype(str))
    
    if not text_data:
        print("Error: Could not find relevant text columns (title, objective, topics) in projects dataset.")
        return

    projects_df['combined_text'] = text_data[0]
    for col_data in text_data[1:]:
        projects_df['combined_text'] += ' ' + col_data

    print("Computing TF-IDF vectors...")
    vectorizer = TfidfVectorizer(stop_words='english')
    # Fit on project texts
    tfidf_matrix = vectorizer.fit_transform(projects_df['combined_text'])
    
    # Transform query
    query_vec = vectorizer.transform([query])
    
    print("Calculating similarity...")
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top matching projects
    top_indices = similarities.argsort()[-n_top:][::-1]
    
    print("\n" + "="*80)
    print(f"Top {n_top} matching projects for query: '{query[:50]}...'")
    print("="*80)
    
    results = []

    for i, idx in enumerate(top_indices):
        project = projects_df.iloc[idx]
        sim_score = similarities[idx]
        project_id = project.get('projectID', project.get('id', 'Unknown'))
        
        title = project.get('title', 'Unknown')
        acronym = project.get('acronym', 'Unknown')
        topics = project.get('topics', 'Unknown')
        start_date = project.get('startDate', 'Unknown')
        end_date = project.get('endDate', 'Unknown')
        
        print(f"\n[{i+1}] Score: {sim_score:.4f} | Project: {title}")
        print(f"    ID: {project_id} | Acronym: {acronym}")
        print(f"    Duration: {start_date} to {end_date}")
        print(f"    Topics: {topics}")
        
        # Find coordinators for this project
        project_orgs = orgs_df[orgs_df['projectID'] == project_id]
        if 'role' in project_orgs.columns:
            coordinators = project_orgs[project_orgs['role'].str.lower() == 'coordinator']
        else:
            coordinators = pd.DataFrame() # empty if role column missing

        if not coordinators.empty:
            for _, org in coordinators.iterrows():
                print(f"    => COORDINATOR: {org.get('name', 'Unknown')}")
                print(f"       Country: {org.get('country', 'Unknown')} | City: {org.get('city', 'Unknown')}")
                if 'shortName' in org and pd.notna(org['shortName']):
                     print(f"       Short Name: {org['shortName']}")
                     
                results.append({
                    'Similarity Score': round(sim_score, 4),
                    'Project ID': project_id,
                    'Project Acronym': acronym,
                    'Project Title': title,
                    'Start Date': start_date,
                    'End Date': end_date,
                    'Coordinator Name': org.get('name', 'Unknown'),
                    'Coordinator Country': org.get('country', 'Unknown'),
                    'Coordinator City': org.get('city', 'Unknown')
                })
        else:
            print("    => COORDINATOR: None found in organization data.")
            results.append({
                'Similarity Score': round(sim_score, 4),
                'Project ID': project_id,
                'Project Acronym': acronym,
                'Project Title': title,
                'Start Date': start_date,
                'End Date': end_date,
                'Coordinator Name': 'None found',
                'Coordinator Country': '',
                'Coordinator City': ''
            })
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Horizon Europe coordinators matching a future call description.")
    parser.add_argument("query", type=str, nargs='?', default=None, help="Text description or keywords of the future call.")
    parser.add_argument("--top", type=int, default=10, help="Number of top projects to return.")
    parser.add_argument("--csv", type=str, help="Optional CSV filename to save the results.")
    try:
        args = parser.parse_args()
        query = args.query
        
        if not query:
            print("="*80)
            print("Horizon Europe Coordinator Finder")
            print("="*80)
            query = input("\nPlease enter the keywords or description for your Horizon call:\n> ")
            
        if query and query.strip():
            results = find_coordinators(query.strip(), args.top)
            if args.csv and results:
                pd.DataFrame(results).to_csv(args.csv, index=False)
                print(f"\nResults saved to {args.csv}")
        else:
            print("No query provided.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        input("\nPress Enter to exit...")
