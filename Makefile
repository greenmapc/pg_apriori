EXTENSION = pg_apriori         # the extensions name
DATA = pg_apriori--1.0.0.sql  # script files to install

# postgres build stuff
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
