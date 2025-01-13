# Local PostgreSQL Setup Guide

This guide will help you set up a local PostgreSQL database, dump the source db and restore it to the destination db.

## Prerequisites

- Ensure you have [Homebrew](https://brew.sh/) installed on your system.

## Installation Steps

1. **Verify Source and Destination Credentials:**
   - Check the `seed_data_from_source.sh` script to ensure the source and destination database credentials are correct.
     - Source credentials: check 1Password for the credentials (can be either staging or prod).
     - Destination credentials: you can use the defaults in the script.

2. **Install and Set Up PostgreSQL:**
   - Run the `install_local_pg.sh` script to install PostgreSQL and set up the initial configuration.
     ```sh
     ./install_local_pg.sh
     ```
  - After the database is set up, you will be asked if you want to seed the local database with data from the source database.
    - If you choose to seed the database, you will be asked to confirm some details and eventually be prompted with the password for the remote source database, which you can find in the `.env` file or on 1Password in the `POSTGRES_PASSWORD` variable.
    - If you choose not to seed the database, you can still run the `seed_data_from_source.sh` script later (next step).

3. **Seed the Local Database:**
   - If Step 2 succeeds, you don't need to run this step as the script would've already run this.
   - If not, run the `seed_data_from_source.sh` script to seed the local database with data from the source database.
     ```sh
     ./seed_data_from_source.sh
     ```
After the seeding is complete, you can view its content using DBeaver or any other PostgreSQL client.

## Notes

- The `install_local_pg.sh` script will:
  - Install PostgreSQL using Homebrew.
  - Install the PGVector extension.
  - Start the PostgreSQL service.
  - Create a default user `postgres` and a default database `yuppdb`.

- The `seed_data_from_source.sh` script will:
  - Prompt for source and destination database details.
  - Export data from the source database.
  - Import data into the local PostgreSQL database.

## Troubleshooting

- Check with bhanu@yupp.ai.

### pgvector issue
During the data restoration time, you may encounter an error related to pgvector. This is because pgvector is not installed on the destination database (your local DB). You need this extension to use vector data types in your DB, which is needed in the `chat_messages` table. Technically `install_local_pg.sh` should have it covered, but if it doesn't work, here is how you can fix it on macOS:

```sh
# use other proper installers in other OS, check official docs
brew install pgvector

# Restart your local db (use the right version)
brew services restart postgresql@16

# Install the pgvector extension in DB, use the right user and DB name though
psql -U postgres -d yuppdb -c "CREATE EXTENSION IF NOT EXISTS pgvector;"
```

The last step might fail and complain not being able to find the file `pgvector.control`. You can fix it this way, and try the last step again.

```sh
# go to the extension folder under the PostgreSQL installation folder
cd /opt/homebrew/opt/postgresql@16/share/postgresql@16/extension/

# make symlinks
ln -s vector.control  pgvector.control
ln -s vector--0.8.0.sql  pgvector--0.8.0.sql
```

## Cleanup

- To uninstall PostgreSQL and remove all related data and configurations, run the `uninstall_local_pg.sh` script.
  ```sh
  ./uninstall_local_pg.sh
  ```