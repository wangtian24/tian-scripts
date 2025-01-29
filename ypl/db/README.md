# General naming convention

The standard for the team is still evolving / being discussed. In the mean time, keep to the existing style, which is based on [sqlstyle.guide](https://www.sqlstyle.guide/#naming-conventions)

# Postgresql migration 

```
db
├── alembic - includes migration script for alembic
│   └── versions - includes migration script history
├── schemaspy - includes schema viewer HTML
```

The environment variables for the database can be found on vercel or in the template file in 1password. 

We use [Alembic](https://alembic.sqlalchemy.org/en/latest/) to manage migrations. 

## Step 0: Make sure everything is up to date at the starting point

1. Update the environment variable in `.env` file to the appropriate value (either local or staging). Change the default values from "changethis" to "changethis1" as the default values will result in errors.
1. Make sure your branch has the latest code from main, then run
```
alembic upgrade head
```
to update your local database to the latest state. Otherwise you may encounter error like 'Target database is not up to date.'

## Step 1: Creating a new migration

1. Change table definiton: update the models in `db/...`.

1. Generate migration script: run the following command to create a new migration:
```bash
alembic revision --autogenerate -m "description for the migration"
```
Note: this does not capture server_default changes. If you change the server_default or adding a new column with a default value, you will need to manually add it to the migration script.

1. After you run the command above, a new file will be created in the `versions` folder. 
1. Review the generated file and make necessary changes. Sometimes you may want to add extra initializations to your table like adding some data.
1. Run `poetry run ruff format ypl/db/` to fix any formatting issues. You can fix any issues iteratively by running `poetry run ruff check --fix`. 
1. Test actual migration: ensure that .env file has the correct environment value (local, staging), and run
```
alembic upgrade head
```
to apply the migration to the corresponding database. 

## Step 2: Review cycle
1. Send the model change and the generated migration script for review.
1. Address any review feedback. After you've made some changes to the model, run `alembic downgrade -1` to downgrade one migration step. 
1. Delete the previous migration script, then repeat `alembic revision --autogenerate` step to create a new migration; apply it with `alembic upgrade head`. 
1. Repeat steps 2 and 3 as necessary.

## Step 3: Submit and update the `sarai-chat` project
1. Once the pull request looks good, submit it.
1. Go to the sarai-chat repository, and run `pnpm prisma:pull`, which updates the `prisma.schema` file to reflect the latest schema in the database
1. Review the auto-generated schema change for prisma. In particular, pay attention to the generated relationships to see if they are correct.
1. Submit the prisma schema file change as another pull request. 

# Schema viewer
## Setup
1. Install DBeaver Community edition. The easiest way to do it is to [install DBeaver](https://dbeaver.io/) Launch the desktop app.
1. Create new connection to the database by using environment variables. For DEV, you can find it in 1Password and for PROD, you can find it in the vercel project.
1. Once connected to the database, you can view the schema by right clicking on the public database or against specific tables and opening the ER Diagram view.