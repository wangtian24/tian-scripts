#!/bin/bash

# Prompt user for confirmation
echo "WARNING: This action will delete all PostgreSQL data and configurations."
read -p "Do you want to proceed? (Y/n): " confirm

# Convert input to lowercase
confirm=${confirm,,}

# Check user's response
if [[ $confirm != "n" && $confirm != "no" ]]; then
    # Existing uninstallation code
    # Stop PostgreSQL service
    echo "Stopping PostgreSQL service..."
    brew services stop postgresql@16

    # Uninstall PostgreSQL
    echo "Uninstalling PostgreSQL..."
    brew uninstall postgresql@16

    # Remove PostgreSQL data directory
    echo "Removing PostgreSQL data directory..."
    rm -rf /opt/homebrew/var/postgresql@16

    # Remove PostgreSQL configuration files
    echo "Removing PostgreSQL configuration files..."
    rm -rf /usr/local/etc/postgresql@16

    echo "PostgreSQL uninstallation complete."
else
    echo "Uninstallation cancelled."
    exit 0
fi
