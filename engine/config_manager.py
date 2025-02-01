import configparser
import os

CONFIG_FILE = "config.ini"
file_Path='config.ini'

def initialize_config():
    """Initialize the config file with default values if it doesn't exist."""
    default_config = {
        "plate_confidence": 85,
        "character_confidence": 75,
        "device": "cuda",
    }
    # If config file doesn't exist, create it
    if not os.path.exists(CONFIG_FILE):
        for key, value in default_config.items():
            save_or_update_config("DEFAULT", key, value)


def save_or_update_config(section, key, value, file_path=file_Path):
    """Save or update a configuration value."""
    config_parser = configparser.ConfigParser()
    config_parser.read(file_path)

    # Handle the DEFAULT section
    if section.upper() == "DEFAULT":
        config_parser["DEFAULT"][key] = str(value)
    else:
        # Ensure the section exists
        if not config_parser.has_section(section):
            config_parser.add_section(section)

        # Update or add the key-value pair in the target section
        config_parser[section][key] = str(value)

    # Save the updated config to the file
    with open(file_path, "w") as config_file:
        config_parser.write(config_file)





def load_config(file_path=file_Path):
    """Load all configurations into a dictionary, excluding inherited DEFAULT keys."""
    config_parser = configparser.ConfigParser()
    config_parser.read(file_path)

    config = {}

    # Store DEFAULT values separately
    config["DEFAULT"] = dict(config_parser["DEFAULT"])

    # Process other sections and exclude inherited keys
    for section in config_parser.sections():
        config[section] = {
            key: value
            for key, value in config_parser[section].items()
            if key not in config["DEFAULT"]
        }

    return config





def add_camera_ip(ip, file_path=CONFIG_FILE):
    """Add a new camera IP dynamically."""
    config_parser = configparser.ConfigParser()
    config_parser.read(file_path)

    # Ensure the CAMERAS section exists
    # if not config_parser.has_section("CAMERAS"):
    #     config_parser.add_section("CAMERAS")
    if not config_parser.has_section("SOURCEDETECT"):
        config_parser.add_section("SOURCEDETECT")

    # Find the next available camera key
    # camera_keys = [key for key in config_parser["CAMERAS"].keys() if key.startswith("camera_")]
    camera_keys = [key for key in config_parser["SOURCEDETECT"].keys() if key.startswith("rtps")]
    next_camera_num = len(camera_keys) + 1
    # new_key = f"camera_{next_camera_num}_ip"
    new_key = f"rtps{next_camera_num}"

    # Add the new camera IP
    config_parser["SOURCEDETECT"][new_key] = ip

    # Save the updated config
    with open(file_path, "w") as config_file:
        config_parser.write(config_file)

    return {new_key: ip}
