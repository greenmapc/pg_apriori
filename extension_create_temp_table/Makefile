EXTENSION = create_temp_table_ext         # the extensions name
DATA = create_temp_table_ext--0.0.1.sql  # script files to install

# postgres build stuff
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
