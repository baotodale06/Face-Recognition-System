import logging

def setup_logging(level: int = logging.INFO,
                  log_to_file: bool = True,
                  filename: str = "app.log") -> None:
    """
    Configure root logger with console and file output

    Args:
        level: Logging level, default: Infor, Warning, Error, Critical
        log_to_file: whether you want to write logs to a file
        filename: file path for logging
    """
    # handler is where logs go, i.e, terminal, file, network, etc.
    handlers: list[logging.Handler] = [logging.StreamHandler()] # the primary one is console/terminal
    if log_to_file:
        handlers.append(logging.FileHandler(filename, encoding='utf-8'))
    
    # set up global (root) logger
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )