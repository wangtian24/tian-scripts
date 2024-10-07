#!/bin/bash

# Install Homebrew if it's not already installed
if ! command -v brew &> /dev/null
then
    echo "***********Homebrew not found. Installing Homebrew...***********"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "***********Homebrew is already installed.***********"
fi

# Install PostgreSQL using Homebrew
echo "***********Installing PostgreSQL...***********"
brew install postgresql@16
brew link postgresql@16

echo "***********Installing PGVector...***********"
curr_dir=$(pwd)
cd /tmp
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
cd $curr_dir #switch back to the original directory

# Start PostgreSQL service
echo "***********Starting PostgreSQL service...***********"
brew services start postgresql@16

# Wait for PostgreSQL to start
sleep 10

# Create default user 'postgres' and default database 'yuppdb'
echo "***********Creating default user 'postgres' and default database 'yuppdb'...***********"
psql postgres -c "CREATE USER postgres WITH PASSWORD 'local';"
psql postgres -c "ALTER USER postgres WITH SUPERUSER;"
psql postgres -c "CREATE DATABASE yuppdb OWNER postgres;"
psql yuppdb -c "CREATE EXTENSION vector;"

echo "***********PostgreSQL installation and setup complete.***********"

# Prompt user to seed the database
read -p "Do you want to seed the local database with data from another source? (Y/n): " seed_choice
seed_choice=${seed_choice:-Y}  # Default to Yes if no input is provided

if [[ $seed_choice =~ ^[Yy]$ ]]
then
    echo "***********Seeding the database...***********"
    DIR="$( cd "$( dirname "$0" )" && pwd )" #get the directory of the current script
    echo $DIR
    bash $DIR/seed_data_from_source.sh
else
    echo "***********Skipping database seeding.***********"
fi

echo "***********Setup process completed.***********"
