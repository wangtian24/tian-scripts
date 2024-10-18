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


3. **Seed the Local Database:**
   - If Step 2 succeeds, you don't need to run this step as the script would've already run this.
   - If not, runthe `seed_data_from_source.sh` script to seed the local database with data from the source database.
     ```sh
     ./seed_data_from_source.sh
     ```

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

## Cleanup

- To uninstall PostgreSQL and remove all related data and configurations, run the `uninstall_local_pg.sh` script.
  ```sh
  ./uninstall_local_pg.sh
  ```