from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from azure.devops.v7_0.work_item_tracking.models import Wiql
import csv
from dotenv import dotenv_values
from bs4 import BeautifulSoup
from typing import List, Tuple, Optional
import pandas as pd

# Constants
CSV_FILE_PATH = 'extracted_bugs_w_dal.csv'
BUG_WORK_ITEM_TYPE = 'Bug'
BUG_STATE_CLOSED = 'Closed'

# Parse env file
config = dotenv_values(".env")

# Connect to Azure DevOps
credentials = BasicAuthentication('', config['personal_access_token'])
connection = Connection(base_url=config['organization_url'], creds=credentials)

class Bug:
    def __init__(self, bug):
        self.bug_id = bug.id
        self.bug_title = bug.fields['System.Title']
        self.priority = bug.fields['Microsoft.VSTS.Common.Priority']
        self.description = parse_html(bug.fields['Microsoft.VSTS.TCM.ReproSteps'])
        self.date = pd.to_datetime(bug.fields['System.CreatedDate'])
        self.commits = []
        self.paths = []

def parse_html(html: str) -> str:
    """Parse HTML content and return plain text."""
    soup = BeautifulSoup(html, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text(separator='\n', strip=True)

def fetch_bugs():
    """Fetch bugs from Azure DevOps."""
    wit_client = connection.clients.get_work_item_tracking_client()
    wiql = Wiql(
        query=f"""
        SELECT * 
        FROM WorkItems
        WHERE [System.WorkItemType] = '{BUG_WORK_ITEM_TYPE}'
        AND [System.AreaPath] UNDER '{config['team_path']}'
        AND [System.State] = '{BUG_STATE_CLOSED}'"""
    )
    query_results = wit_client.query_by_wiql(wiql).work_items
    bugs = []

    print("Starting bug extraction...")
    if query_results:
        for item in query_results:
            bug = wit_client.get_work_item(item.id, expand="Relations")
            bug_obj = Bug(bug)
            process_bug_relations(bug, bug_obj)
            bugs.append(bug_obj)
    print("Bug extraction complete.")
    return bugs

def normalize_path(path: str) -> str:
    """Normalize file paths."""
    path = path.replace("\\", "/")

    if path.startswith("/Source"):
        return path[len('/Source'):]
    return path


def process_bug_relations(bug, bug_obj: Bug):

    toIgnore = ['Resources', 'png', 'rspakspec','test', 'Test', 'yml', 'Build', '/Core', '/Build/', '.Build.', '/MessageLib', '/packages', 'csproj', 'sln', 'cmd', 'md', '/ModelLib', '/SCIDLib', 'Settings']

    """Process relations for a bug and fetch associated commits."""
    if bug.relations:
        seen_paths = set()
        for relation in bug.relations:
            if "Commit" in relation.url:
                commit_id = extract_commit_id(relation.url)
                if commit_id:
                    commit, paths = fetch_commit(commit_id)
                    if commit:
                        bug_obj.commits.append(commit)
                        for path in paths:
                            # skip test files and duplicates and other non relevant stuff
                            if any(ignore in path for ignore in toIgnore):
                                continue
                            norm_path = normalize_path(path)
                            if norm_path not in seen_paths:
                                seen_paths.add(norm_path)
                                bug_obj.paths.append(norm_path)
                        print(f"Appending commit to bug: {bug.id}")

def extract_commit_id(url: str) -> Optional[str]:
    """Extract commit ID from a URL."""
    try:
        return url.split('/')[-1].split('%')[-1][2:]
    except IndexError:
        print(f"Failed to extract commit ID from URL: {url}")
        return None

def fetch_commit(commit_id: str) -> Tuple[Optional[dict], List[str]]:
    """Fetch commit details and associated file paths."""
    git_client = connection.clients.get_git_client()
    try:
        commit = git_client.get_commit(commit_id, config['team'], config['project'])
        changes = get_commit_changes(commit_id)
        paths = [c['item']['path'] for c in changes.changes if c['item']['gitObjectType'] == 'blob']
        return commit, paths
    except Exception as e:
        print(f"Error fetching commit {commit_id}: {e}")
        return None, []

def get_commit_changes(commit_id: str) -> Optional[dict]:
    """Fetch changes for a specific commit."""
    git_client = connection.clients.get_git_client()
    try:
        return git_client.get_changes(commit_id, config['team'], config['project'])
    except Exception as e:
        print(f"Error fetching changes for commit {commit_id}: {e}")
        return None

def create_csv(bugs: List[Bug]) -> None:
    """Create a CSV file with bug data."""
    data = [['Date','Bug ID', 'Bug Title', 'Priority', 'Description', 'Paths']]
    data.extend(
        [b.date, b.bug_id, b.bug_title, b.priority, b.description, b.paths]
        for b in bugs if b.paths
    )
    with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"CSV created successfully at {CSV_FILE_PATH}.")

def main():
    bugs = fetch_bugs()
    create_csv(bugs)

if __name__ == "__main__":
    main()