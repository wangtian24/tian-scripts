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
3. Delete the previous migration script, then repeat `alembic revision --autogenerate` step to crete a new migration; apply it with `alembic upgrade head`. 
4. Repeat steps 2 and 3 as necessary.

## Step 3: Submit and update the `sarai-chat` project
1. Once the pull request looks good, submit it.
2. Go to the sarai-chat repository, and run `pnpm prisma:pull`, which updates the `prisma.schema` file to reflect the latest schema in the database
3. Review the auto-generated schema change for prisma. In particular, pay attention to the generated relationships to see if they are correct.
4. Submit the prisma schema file change as another pull request. 

# Generating schema spy ER diagrams and schema viewer HTML
## Setup
1. Install docker engine. The easiest way to do it is to [install Docker Desktop](https://docs.docker.com/desktop/install/mac-install/) which comes with a graphical interface as well as Docker Engine (CLI + Daemon). Launch the desktop app to complete the setup.
2. Make a copy of `scheamspy.properties.sample` and rename it to `schemaspy.properties`. Update the database password under `schemaspy.p`. It should already be in your top level `.env` file, or find it in 1password if you never set that up.

### Generating new assets
1. Here we are assuming that you are running the commands at the root directory of the `yupp-llms` repo. Adjust the paths below where `$PWD` was referenced accordingly if you are running the command from elsewhere.
2. Run the following command. Docker will pull the image when you run it for the first time. 
   ```
   docker run  -v "$PWD/db/schemaspy:/output" -v "$PWD/db/schemaspy/schemaspy.properties:/schemaspy.properties" schemaspy/schemaspy:snapshot
   ```
3. You can view the output by opening the [index.html](schemaspy/index.html) file in your browser.
