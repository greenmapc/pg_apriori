EXTENSION = test         # the extensions name
DATA = test--0.0.1.sql  # script files to install

REGRESS = test

# postgres build stuff
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
