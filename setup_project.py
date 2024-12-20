from pathlib import Path
import os

def create_directory_structure():
    # Get the current directory
    base_dir = Path.cwd()
    
    # Create main directories
    directories = [
        'data_manager/data',
        'garch/models',
        'ensemble_stats',
        'regression',
        'strategy',
        'results',
        'tests',
        'config',
        'notebooks'
    ]
    
    for dir_path in directories:
        Path(base_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
    # Create __init__.py files in all package directories
    package_dirs = [
        'data_manager',
        'garch',
        'ensemble_stats',
        'regression',
        'strategy',
        'config',
        'tests'
    ]
    
    for dir_path in package_dirs:
        init_file = Path(base_dir / dir_path / '__init__.py')
        init_file.touch(exist_ok=True)
    
    # Create main execution scripts
    main_scripts = [
        'setup_database.py',
        'calculate_historical.py',
        'update_analysis.py',
        'run_strategy.py'
    ]
    
    for script in main_scripts:
        Path(base_dir / script).touch(exist_ok=True)
    
    # Create configuration files
    config_files = [
        'config/database_config.py',
        'config/model_config.py',
        'config/strategy_config.py',
        '.env.example'
    ]
    
    for config_file in config_files:
        Path(base_dir / config_file).touch(exist_ok=True)
    
    # Create core module files
    module_files = {
        'data_manager': ['database.py', 'data_loader.py', 'holiday_handler.py'],
        'garch': ['estimator.py', 'forecaster.py', 'parallel_processor.py'],
        'ensemble_stats': ['calculator.py', 'statistics.py'],
        'regression': ['expander.py', 'error_correction.py'],
        'strategy': ['backtester.py', 'portfolio.py', 'performance.py']
    }
    
    for module, files in module_files.items():
        for file in files:
            Path(base_dir / module / file).touch(exist_ok=True)
    
    # Create test files
    test_files = [
        'test_data_manager.py',
        'test_garch.py',
        'test_ensemble.py',
        'test_regression.py',
        'test_strategy.py'
    ]
    
    for test_file in test_files:
        Path(base_dir / 'tests' / test_file).touch(exist_ok=True)
    
    # Create and populate requirements.txt
    requirements = [
        'numpy',
        'pandas',
        'scipy',
        'arch',
        'statsmodels',
        'duckdb',
        'pytest',
        'python-dotenv',
        'jupyterlab',
        'black',
        'flake8',
        'mypy'
    ]
    
    with open(base_dir / 'requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    
    # Create .gitignore with project-specific ignores
    gitignore_content = """# Project specific ignores
.env
*.db
data_manager/data/*
results/*
*.pyc
__pycache__/
.pytest_cache/
.coverage
htmlcov/
"""
    
    with open(base_dir / '.gitignore', 'w') as f:
        f.write(gitignore_content)

def main():
    print("Setting up project directory structure...")
    create_directory_structure()
    print("Project setup complete!")
    print("\nDirectory structure created:")
    
    # Print directory tree
    def print_tree(directory, prefix=""):
        paths = sorted(Path(directory).glob("*"))
        for i, path in enumerate(paths):
            is_last = i == len(paths) - 1
            print(f"{prefix}{'└── ' if is_last else '├── '}{path.name}")
            if path.is_dir() and path.name not in ['.git', '__pycache__']:
                print_tree(path, prefix + ('    ' if is_last else '│   '))
    
    print_tree(Path.cwd())

if __name__ == "__main__":
    main()