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

## Step 1: Creating a new migration

1. Change table definiton: update the models in `db/...`. 
2. Generate migration script: run the following command to create a new migration:
```bash
alembic revision --autogenerate -m "description for the migration"
```
3. A new file will be created in the `versions` folder. 
4. Review the generated file and make necessary changes. 
5. Test actual migration: ensure that .env file has the correct environment value, and run `alembic upgrade head` to apply the migration to the corresponding database. 

## Step 2: Review cycle
1. Send the model change and the generated migration script for review.
2. Address any review feedback. After you've made some changes to the model, run `alembic downgrade -1` to downgrade one migration step. 
3. Delete the previous migration script, then repeat `alembic revision --autogenerate` step to create a new migration; apply it with `alembic upgrade head`. 
4. Repeat steps 2 and 3 as necessary.

## Step 3: Submit and update the `sarai-chat` project
1. Once the pull request looks good, submit it.
2. Go to the sarai-chat repository, and run `pnpm prisma:pull`, which updates the `prisma.schema` file to reflect the latest schema in the database
3. Review the auto-generated schema change for prisma. In particular, pay attention to the generated relationships to see if they are correct.
4. Submit the prisma schema file change as another pull request. 

# Schema viewer
## Setup
1. Install DBeaver Community edition. The easiest way to do it is to [install DBeaver](https://dbeaver.io/) Launch the desktop app.
2. Create new connection to the database by using environment variables. For DEV, you can find it in 1Password and for PROD, you can find it in the vercel project.
3. Once connected to the database, you can view the schema by right clicking on the public database or against specific tables and opening the ER Diagram view.