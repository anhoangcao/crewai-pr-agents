from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import GithubSearchTool
from github import Github
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

@CrewBase
class RagAssistant:
    """RagAssistant crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, github_url=None, content_types=None):
        self.github_token = os.getenv('GITHUB_TOKEN')
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        self.github_url = github_url
        self.content_types = content_types
        self.github = Github(self.github_token)

    def create_github_tool(self):
        """Create a GitHub search tool."""
        return GithubSearchTool(
            github_repo=self.github_url,
            gh_token=self.github_token,
            content_types=self.content_types,
            config=dict(
                llm=dict(
                    provider="ollama",
                    config=dict(
                        model="llama2",
                    ),
                ),
                embedder=dict(
                    provider="ollama",
                    config=dict(
                        model="deepseek-coder",
                    ),
                )
            )
        )

    def extract_pr_info(self, pr_url):
        """Extract owner, repo, and PR number from GitHub PR URL."""
        pattern = r"(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+)/pull/(\d+)"
        match = re.search(pattern, pr_url)
        if not match:
            raise ValueError(f"Invalid GitHub PR URL format. URL provided: {pr_url}")
        owner, repo_name, pr_number = match.groups()
        print(f"Analyzing PR: {owner}/{repo_name}#{pr_number}")
        return owner, repo_name, pr_number

    def get_pr_files(self, pr):
        """Get list of files changed in PR with their stats."""
        files_info = []
        for file in pr.get_files():
            files_info.append({
                'filename': file.filename,
                'additions': file.additions,
                'deletions': file.deletions,
                'changes': file.changes,
                'status': file.status
            })
        return files_info

    def describe_pull_request(self):
        """Analyze and describe a specific GitHub Pull Request."""
        try:
            # Extract PR information
            owner, repo_name, pr_number = self.extract_pr_info(self.github_url)
            print(f"Fetching PR details...")

            # Get repository and PR
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            pr = repo.get_pull(int(pr_number))

            # Get files information
            files_info = self.get_pr_files(pr)

            # Format files information
            files_summary = "\n".join([
                f"- {file['filename']} ({file['additions']} additions, {file['deletions']} deletions)"
                for file in files_info
            ])

            # Create a special PR analysis agent
            pr_analyst = Agent(
                name="PR Analyst",
                role="Pull Request Analysis Expert",
                goal="Analyze GitHub Pull Requests and provide comprehensive descriptions",
                backstory="""You are an expert at analyzing GitHub Pull Requests. 
                You understand code changes, commit patterns, and can provide detailed 
                yet concise descriptions of changes.""",
                verbose=True,
                llm=LLM(model="ollama/llama2", base_url="http://localhost:11434")
            )

            # Create analysis task
            analysis_task = Task(
                description=f"""Analyze PR #{pr_number} in {owner}/{repo_name}

    Title: {pr.title}
    Author: {pr.user.login}
    Status: {'Merged' if pr.is_merged() else 'Open' if pr.state == 'open' else 'Closed'}
    Created: {pr.created_at}
    Last Updated: {pr.updated_at}

    Description:
    {pr.body}

    Changes Overview:
    - Total Files Changed: {pr.changed_files}
    - Additions: {pr.additions}
    - Deletions: {pr.deletions}

    Modified Files:
    {files_summary}

    Please provide a comprehensive analysis including:
    1. Overview of Changes
    2. Technical Impact Assessment
    3. Code Analysis
    4. Risk Assessment
    5. Recommendations
                """,
                agent=pr_analyst,
                expected_output="A detailed analysis of the pull request with technical insights and recommendations."
            )

            # Create and run the crew
            crew = Crew(
                agents=[pr_analyst],
                tasks=[analysis_task],
                process=Process.sequential,
                verbose=True
            )

            # Run the analysis
            print("\nAnalyzing PR contents...")
            result = crew.kickoff()
            print("\nPull Request Analysis:")
            print("-" * 50)
            print(result)
            return result

        except Exception as e:
            import traceback
            print(f"Full error traceback:")
            print(traceback.format_exc())
            raise Exception(f"Error analyzing Pull Request: {str(e)}")

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            llm=LLM(model="ollama/llama2", base_url="http://localhost:11434"),
            tools=[self.create_github_tool()]
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            llm=LLM(model="ollama/llama2", base_url="http://localhost:11434")
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            expected_output="Find relevant research materials and summarize findings."
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            expected_output="Generate a detailed report in Markdown format.",
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RagAssistant crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )