import click
from .crew import RagAssistant
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

@click.group()
def cli():
    """RAG Assistant CLI - A tool for AI-powered research and analysis"""
    pass

@cli.command()
@click.option('--topic', '-t', required=True, help='Topic to research')
@click.option('--github-url', '-g', required=True, help='GitHub repository URL')
@click.option('--content-types', '-c', multiple=True, 
              type=click.Choice(['code', 'repo', 'pr', 'issue'], case_sensitive=False),
              default=['code', 'pr'], help='Content types to analyze')
def run(topic, github_url, content_types):
    """Run the RAG Assistant with specified parameters"""
    try:
        inputs = {
            'topic': topic
        }
        rag_assistant = RagAssistant(github_url=github_url, content_types=list(content_types))
        rag_assistant.crew().kickoff(inputs=inputs)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)

@cli.command()
@click.option('--pr-url', '-u', required=True, help='GitHub Pull Request URL')
def describe_pr(pr_url):
    """Analyze and describe a specific GitHub Pull Request"""
    try:
        rag_assistant = RagAssistant(github_url=pr_url, content_types=['pr'])
        rag_assistant.describe_pull_request()
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)

@cli.command()
@click.argument('iterations', type=int)
@click.argument('filename')
@click.option('--topic', '-t', required=True, help='Topic for training')
@click.option('--github-url', '-g', required=True, help='GitHub repository URL')
@click.option('--content-types', '-c', multiple=True, 
              type=click.Choice(['code', 'repo', 'pr', 'issue'], case_sensitive=False),
              default=['code', 'pr'], help='Content types to analyze')
def train(iterations, filename, topic, github_url, content_types):
    """Train the crew for a specified number of iterations"""
    try:
        inputs = {
            'topic': topic
        }
        rag_assistant = RagAssistant(github_url=github_url, content_types=list(content_types))
        rag_assistant.crew().train(n_iterations=iterations, filename=filename, inputs=inputs)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)

@cli.command()
@click.argument('task_id')
def replay(task_id):
    """Replay a specific task execution"""
    try:
        rag_assistant = RagAssistant()
        rag_assistant.crew().replay(task_id=task_id)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)

@cli.command()
@click.argument('iterations', type=int)
@click.argument('model_name')
@click.option('--topic', '-t', required=True, help='Topic for testing')
@click.option('--github-url', '-g', required=True, help='GitHub repository URL')
@click.option('--content-types', '-c', multiple=True, 
              type=click.Choice(['code', 'repo', 'pr', 'issue'], case_sensitive=False),
              default=['code', 'pr'], help='Content types to analyze')
def test(iterations, model_name, topic, github_url, content_types):
    """Test the crew with specified iterations and model"""
    try:
        inputs = {
            'topic': topic
        }
        rag_assistant = RagAssistant(github_url=github_url, content_types=list(content_types))
        rag_assistant.crew().test(n_iterations=iterations, openai_model_name=model_name, inputs=inputs)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)

if __name__ == '__main__':
    cli()