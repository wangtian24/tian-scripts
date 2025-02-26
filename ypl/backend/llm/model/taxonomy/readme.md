# Model Taxonomy Management

SQL scripts for model taxonomy manual maintenance. Pleaese only run them when you are absolutely sure you understand what they do. Check [this Superset Dashboard](https://superset.yupp.ai/superset/dashboard/33/?native_filters_key=9TcGZUH9uz7x0UsF4EraYul-rmYPR4BmOORqRISMJsqt2peMvNFRo4ScnCULT214) for the current model and taxonomy data.



## LanguageModel <> LanguageModelTaxonomy linkages

Every entry in the `language_models` (LM) table represent one provider-specific model, it should have a corresponding entry in the  `language_model_taxonomy` (LMT) table through `taxonomy_id` field, which corresponds to a node in the model taxonomy tree. Sometimes this linkage might be broken or missing. Here are the steps to fix it. Use this with caution and please consult Tian before running these on production or staging.

### STEP 0: Fix Model Publishers
Find all entries missing the model_publisher info, fill in values yourself. Check [Superset Dashboard](https://superset.yupp.ai/superset/dashboard/33/?native_filters_key=9TcGZUH9uz7x0UsF4EraYul-rmYPR4BmOORqRISMJsqt2peMvNFRo4ScnCULT214) for example values.
```sql
select 
  lm.name,
  lm.model_publisher,
from language_models lm
where lm.model_publisher is null;
```
### STEP 1: Pull from LMT (update only)
If LM entries already have taxonomy data but just no taxonomy_id, this scripts establish such a linkage if corresponding rows could be found in LMT.
```sql
update language_models lm
set taxonomy_id = lmt.language_model_taxonomy_id
from language_model_taxonomy lmt 
where
  lm.model_publisher = lmt.model_publisher
  and lm.family = lmt.model_family
  and (lm.model_class = lmt.model_class or (lm.model_class is null and lmt.model_class is null))
  and (lm.model_version = lmt.model_version or (lm.model_version is null and lmt.model_version is null))
  and (lm.model_release = lmt.model_release or (lm.model_release is null and lmt.model_release is null))
  and lm.taxonomy_id is null
```

### STEP 2: Push to LMT (update only)
If no LMT entries could be found with the matching value, we create new LMT entry based on the values in LM table.
There might be new rows created afterwards, if so, go to LMT table and check all rows with `taxo_label` field starting with "NEW:" and edit their names so they look good.
```sql
with new_taxonomy as (
  insert into language_model_taxonomy (
    created_at,
    language_model_taxonomy_id,
    taxo_label,
    model_publisher,
    model_family,
    model_class,
    model_version,
    model_release,
    is_pickable,
    is_leaf_node
  )
  select distinct
    now(),
    gen_random_uuid(),
    concat_ws(' ', 'NEW:', lm.family, lm.model_version, lm.model_class),
    lm.model_publisher,
    lm.family,
    lm.model_class,
    lm.model_version,
    lm.model_release,
    true,
    true
  from language_models lm
  where lm.taxonomy_id is null
    and lm.family is not null
    and lm.model_class is not null
  returning language_model_taxonomy_id, model_publisher, model_family, model_class, model_version, model_release
)
update language_models lm
set taxonomy_id = nt.language_model_taxonomy_id
from new_taxonomy nt
where lm.model_publisher = nt.model_publisher
  and lm.family = nt.model_family
  and (lm.model_class = nt.model_class or (lm.model_class is null and nt.model_class is null))
  and (lm.model_version = nt.model_version or (lm.model_version is null and nt.model_version is null))
  and (lm.model_release = nt.model_release or (lm.model_release is null and nt.model_release is null))
  and lm.taxonomy_id is null
```

### STEP 3: taxo_label duplicate check (may need edit)
`taxo_label` is the display label for a model taxonomy entry, there shouldn't be multiple pickable entries in LMT with the same `taxo_label` or they will show up in the model picker as dupes which could be confusing for users. Examine the returned results, go through each group of rows with the same `taxo_label`, and check
1. Two entries can have the same `taxo_label` if they only differ by `model_release`, if not, like they differ by other columns, you need to merge them, especially if it's just a upper/lower case difference. Decide which one you want to keep, and point all corresponding entries in `language_models` to your picked entry via `taxonomy_id`. Similarly, only one entry can have `is_pickable` set to true.
1. set `is_pickable` to true to only one entry in the group, and false to everyone else. This is the entry with the latest release date/marker.

```sql
select 
  lmt.language_model_taxonomy_id,
  lmt.taxo_label,
  lmt.model_publisher,
  lmt.model_family,
  lmt.model_class,
  lmt.model_version,
  lmt.model_release,
  lmt.is_pickable,
  lmt.is_leaf_node,
  count(lm.language_model_id) as model_count
from language_model_taxonomy lmt
left join language_models lm on lm.taxonomy_id = lmt.language_model_taxonomy_id
where lower(replace(lmt.taxo_label, ' ', '')) in (
  select lower(replace(taxo_label, ' ', '')) as normalized_label
  from language_model_taxonomy
  where is_pickable = true
  group by lower(replace(taxo_label, ' ', ''))
  having count(*) > 1
)
group by
  lmt.language_model_taxonomy_id,
  lmt.taxo_label,
  lmt.model_publisher,
  lmt.model_family,
  lmt.model_class,
  lmt.model_version,
  lmt.model_release,
  lmt.is_pickable,
  lmt.is_leaf_node
order by lower(replace(lmt.taxo_label, ' ', ''));
```

### STEP 4: Pickability Check (may need edit)
This step checks if any active model in LM links to a non-pickable LMT entry, it means this model cannot be picked in taxonomy-based picker. Examine the results, for every `taxo_label` every group, make sure:
-- 1. there's at most one pickable entry, if not, set `is_pickable` to true on one entry and false to everyone else.
-- 2. if it's a leaf node, it must link to a LM entry (`language_model_id` is not null)
```sql
select 
  lmt.taxo_label,
  lmt.model_publisher,
  lmt.model_family,
  lmt.model_class,
  lmt.model_version,
  lmt.model_release,
  lmt.is_pickable,
  lmt.is_leaf_node,
  lmt.language_model_taxonomy_id as taxo_id,
  lm.name,
  lm.internal_name,
  lm.language_model_id,
  lm.status
from language_model_taxonomy lmt
left join language_models lm on lm.taxonomy_id = lmt.language_model_taxonomy_id
where lmt.taxo_label in (
  select distinct lmt.taxo_label
  from language_models lm
  left join language_model_taxonomy lmt on lm.taxonomy_id = lmt.language_model_taxonomy_id
  where lm.taxonomy_id is not null
    and lm.deleted_at is null
    and lm.status = 'ACTIVE'
    and lmt.is_pickable = false
)
  and lm.deleted_at is null
order by lmt.taxo_label, lm.name
```
### STEP 5: Pickability Confirmation (no edit needed)
A final check to make sure every LMT `taxo_label` only corresponds to one pickable entry. If this shows any result, repeat the last step.
```sql
select 
  lmt.taxo_label,
  count(*) as total_entries,
  sum(case when is_pickable then 1 else 0 end) as pickable_count
from language_model_taxonomy lmt
group by lmt.taxo_label
having sum(case when is_pickable then 1 else 0 end) > 1
order by lmt.taxo_label
```

### STEP 6: Other Missing Fields Check (may need edit)
Make sure all pickable entries have proper context window and avatar url. Copy info from LM to LMT if LMT is null, or find the information yourself.
```sql
select 
  lmt.taxo_label,
  lmt.model_publisher,
  lmt.model_family,
  lmt.model_class,
  lmt.model_version,
  lmt.model_release,
  lmt.context_window_tokens as taxo_context_window,
  lm.context_window_tokens as model_context_window,
  lmt.avatar_url as taxo_avatar_url,
  lm.avatar_url as model_avatar_url,
  lmt.language_model_taxonomy_id as taxo_id,  
  lm.language_model_id as model_id
from language_model_taxonomy lmt
left join language_models lm on lm.taxonomy_id = lmt.language_model_taxonomy_id
where lmt.is_pickable = true
  and (
    lmt.context_window_tokens is null 
    or lmt.context_window_tokens = 0
    or lmt.avatar_url is null
    or trim(lmt.avatar_url) = ''
  )
order by lmt.taxo_label;
```

Now the LM and LMT tables should be in a better shape.

TODO: We will add some checks to maintenance script and run them in cron jobs.

## Formating field values

Here are some other convenience SQLs for formatting field values, usually you don't need to run them.

```sql
-- update model publisher
update language_models lm
set model_publisher = lmt.model_publisher
from language_model_taxonomy lmt 
where lm.taxonomy_id  = lmt.language_model_taxonomy_id
  and lm.taxonomy_id  is not null

-- Set all model_class to lower case in LMT
update language_model_taxonomy
set model_class = lower(model_class);

-- Set all model_class to lower case in LM
update language_models
set model_class = lower(model_class);
```
