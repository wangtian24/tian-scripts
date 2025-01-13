#!/bin/bash
set -e

echo "This script will dump the source database and restore it to a local destination database."
echo
# Default values point to the GCP Development DB fronted with PgBouncer
DEFAULT_SRC_HOST="34.85.174.70" #pgbouncer
DEFAULT_SRC_PORT="64321" #pgbouncer port
DEFAULT_SRC_DB="yuppdb"
DEFAULT_SRC_USER="be_app_user"

# Default values point to the local development DB
DEFAULT_DEST_HOST="localhost"
DEFAULT_DEST_PORT="5432"
DEFAULT_DEST_DB="yuppdb"
DEFAULT_DEST_USER="postgres"
DEFAULT_DEST_PASSWORD="local" # default pwd for local pg installed through script

# Function to prompt for input with default value
prompt_with_default() {
    local prompt="$1"
    local default="$2"
    local input

    read -p "$prompt [$default]: " input
    echo "${input:-$default}"
}

# Check if users wants to use a previous dump file
DUMP_FILE=$(prompt_with_default "Press ENTER to create a new dump, or enter the earlier dump file path (find in /tmp/db_dump_*): " "")
if [ -z "$DUMP_FILE" ]; then
    echo "Creating a new dump"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    DUMP_FILE="/tmp/db_dump_${TIMESTAMP}.backup"

    # Source database details
    SRC_HOST=$(prompt_with_default "Enter source host" "$DEFAULT_SRC_HOST")
    SRC_PORT=$(prompt_with_default "Enter source port" "$DEFAULT_SRC_PORT")
    SRC_DB=$(prompt_with_default "Enter source database name" "$DEFAULT_SRC_DB")
    SRC_USER=$(prompt_with_default "Enter source username" "$DEFAULT_SRC_USER")
    #SRC_PASSWORD=$(prompt_with_default "Enter source password" "$DEFAULT_SRC_PASSWORD")

    echo "--- Start dumping source database, this will take a few minutes..."
    pg_dump -h $SRC_HOST -p $SRC_PORT -U $SRC_USER -d $SRC_DB -F c -b -v -f $DUMP_FILE

    if [ $? -ne 0 ]; then
        echo "--- Error: database dumping failed"
        exit 1
    fi
    echo "--- Database dump successful, the dump file is at $DUMP_FILE"
else
    echo "--- Using existing dump file $DUMP_FILE"
fi


# Destination database details
echo "--- Now enter the information for the destination (local) database to restore to:"
DEST_HOST=$(prompt_with_default "Enter destination host" "$DEFAULT_DEST_HOST")
DEST_PORT=$(prompt_with_default "Enter destination port" "$DEFAULT_DEST_PORT")
DEST_DB=$(prompt_with_default "Enter destination database name" "$DEFAULT_DEST_DB")
DEST_USER=$(prompt_with_default "Enter destination username" "$DEFAULT_DEST_USER")
DEST_PASSWORD=$(prompt_with_default "Enter destination password" "$DEFAULT_DEST_PASSWORD")


DROP_DB=$(prompt_with_default "Drop and re-create database before restoring? (${DEST_HOST}:${DEST_PORT}/${DEST_DB})? (y/N)" "N")
if [[ $DROP_DB =~ ^[Yy]$ ]]; then
    echo "--- Dropping and re-creating database"
    PGPASSWORD=$DEST_PASSWORD dropdb -h $DEST_HOST -p $DEST_PORT -U $DEST_USER $DEST_DB
    PGPASSWORD=$DEST_PASSWORD createdb -h $DEST_HOST -p $DEST_PORT -U $DEST_USER $DEST_DB
else
    echo "--- Not dropping existing database"
fi

echo "--- Restoring to destination database"
PGPASSWORD=$DEST_PASSWORD pg_restore -h $DEST_HOST -p $DEST_PORT -U $DEST_USER -d $DEST_DB -v $DUMP_FILE --no-owner --no-privileges

if [ $? -ne 0 ]; then
    echo "--- Restoration failed"
    exit 1
fi

DELETE_DUMP=$(prompt_with_default "Remove dump file? (Y/n)" "Y")
if [[ $DELETE_DUMP =~ ^[Yy]$ ]]; then
    echo "--- Removing dump file $DUMP_FILE"
    # rm $DUMP_FILE
else
    echo "--- Not removing dump file, you can use it later"
fi

echo "--- Database seeding completed successfully"
