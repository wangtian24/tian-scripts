#!/bin/bash

# Default values point to the GCP Development DB fronted with PgBouncer
DEFAULT_SRC_HOST="34.85.174.70" #pgbouncer
DEFAULT_SRC_PORT="6432" #pgbouncer port
DEFAULT_SRC_DB="yuppdb"
DEFAULT_SRC_USER="postgres"
DEFAULT_SRC_PASSWORD="" #user input only

# Default values point to the local development DB
DEFAULT_DEST_HOST="localhost"
DEFAULT_DEST_PORT="5432"
DEFAULT_DEST_DB="yuppdb"
DEFAULT_DEST_USER="postgres"
DEFAULT_DEST_PASSWORD="local"

# Function to prompt for input with default value
prompt_with_default() {
    local prompt="$1"
    local default="$2"
    local input

    read -p "$prompt [$default]: " input
    echo "${input:-$default}"
}

# Source database details
SRC_HOST=$(prompt_with_default "Enter source host" "$DEFAULT_SRC_HOST")
SRC_PORT=$(prompt_with_default "Enter source port" "$DEFAULT_SRC_PORT")
SRC_DB=$(prompt_with_default "Enter source database name" "$DEFAULT_SRC_DB")
SRC_USER=$(prompt_with_default "Enter source username" "$DEFAULT_SRC_USER")
SRC_PASSWORD=$(prompt_with_default "Enter source password" "$DEFAULT_SRC_PASSWORD")

# Destination database details
DEST_HOST=$(prompt_with_default "Enter destination host" "$DEFAULT_DEST_HOST")
DEST_PORT=$(prompt_with_default "Enter destination port" "$DEFAULT_DEST_PORT")
DEST_DB=$(prompt_with_default "Enter destination database name" "$DEFAULT_DEST_DB")
DEST_USER=$(prompt_with_default "Enter destination username" "$DEFAULT_DEST_USER")
DEST_PASSWORD=$(prompt_with_default "Enter destination password" "$DEFAULT_DEST_PASSWORD")

# Temporary file for the dump
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DUMP_FILE="/tmp/db_dump_${TIMESTAMP}.backup"

echo "***********Backing up source database...***********"
PGPASSWORD=$SRC_PASSWORD pg_dump -h $SRC_HOST -p $SRC_PORT -U $SRC_USER -d $SRC_DB -F c -b -v -f $DUMP_FILE

if [ $? -ne 0 ]; then
    echo "***********Error: Backup failed***********"
    exit 1
fi

echo "***********Restoring to destination database...***********"
PGPASSWORD=$DEST_PASSWORD pg_restore -h $DEST_HOST -p $DEST_PORT -U $DEST_USER -d $DEST_DB -v $DUMP_FILE --no-owner --no-privileges

if [ $? -ne 0 ]; then
    echo "***********Error: Restore failed***********"
    exit 1
fi

echo "***********Cleaning up...***********"
rm $DUMP_FILE

echo "***********Database migration completed successfully.***********"
