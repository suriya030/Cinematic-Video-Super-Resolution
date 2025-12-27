import os
import torch
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

def get_device(device_preference='auto'):
    """Get the appropriate device for PyTorch operations"""
    if device_preference == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_preference)
    
    print(f"{Fore.BLUE}🔧 Initializing IQA models on device: {Fore.GREEN}{device}")
    return device

def print_header(title):
    """Print a beautifully formatted header with centered text and decorative elements"""
    header_width = 80
    
    # Calculate padding to center the title with sparkles
    title_with_sparkles = f" {title} "
    title_length = len(title_with_sparkles)
    
    # Handle cases where title might be too long
    if title_length >= header_width - 4:
        padding_left = 2
        padding_right = max(0, header_width - title_length - 2)
    else:
        padding_left = (header_width - title_length - 2) // 2
        padding_right = header_width - title_length - padding_left - 2
    
    # Print the formatted header
    print(f"\n{Fore.MAGENTA}{'━' * header_width}")
    print(f"{Fore.MAGENTA}┃{' ' * padding_left}{Fore.BLUE}{title_with_sparkles}{Fore.MAGENTA}{' ' * padding_right}┃")
    print(f"{Fore.MAGENTA}{'━' * header_width}{Style.RESET_ALL}\n")

def print_step(step_num, description):
    """Print a formatted step header"""
    print(f"\n{Fore.CYAN}\033[1m STEP {step_num}: {description}\033[0m")

def print_success(message):
    """Print a success message"""
    print(f"{Fore.GREEN}✅ {message}")

def print_warning(message):
    """Print a warning message"""
    print(f"{Fore.YELLOW}⚠️ {message}")

def print_info(message):
    """Print an info message"""
    print(f"{Fore.BLUE} {message}")

def print_processing(message):
    """Print a processing message"""
    print(f"{Fore.MAGENTA} {message}")

def get_base_filename(file_path):
    """Extract base filename without extension"""
    return os.path.basename(file_path).split('.')[0]

def ensure_directory(file_path):
    """Ensure directory exists for given file path"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)