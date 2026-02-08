import logging
import datetime
from pathlib import Path


def setup_logger(name, env_name, timestamp=None, frame_skip=4, sticky_action_p=0.0,
                 horizon=400, optim_steps=10, memory_size=5, log_dir=None,
                 enable_console=True, enable_file=True, additional_params=None, prefix=None,
                 log_file=None):
    """
    Set up a standardized logger with console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        env_name: Environment name (e.g., "PongNoFrameskip-v4")
        timestamp: Timestamp string (if None, will be generated)
        frame_skip: Frame skip value for log file naming
        sticky_action_p: Sticky action probability for log file naming
        horizon: Horizon value for log file naming
        optim_steps: Number of optimization steps for log file naming
        memory_size: Memory size for log file naming
        log_dir: Directory for log files (defaults to "logs")
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        additional_params: Dictionary of additional parameters to include in filename
        prefix: Optional prefix string to add before timestamp (e.g., "OCAtari")
        log_file: Optional explicit log file path; when provided, used directly instead of constructing a filename

    Returns:
        Configured logger instance
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        if log_file is None:
            # Build filename from parameters
            filename_parts = [env_name.replace('/', '_')]
            if prefix:
                filename_parts.append(prefix)
            if additional_params:
                for key, value in additional_params.items():
                    filename_parts.append(f"{key}{value}")
            else:
                filename_parts.extend([
                    timestamp,
                    f"skip{frame_skip}",
                    f"sticky{sticky_action_p}",
                    f"horizon{horizon}",
                    f"optimSteps{optim_steps}",
                    f"mem{memory_size}"
                ])
            log_file = log_dir / f"{'_'.join(filename_parts)}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
