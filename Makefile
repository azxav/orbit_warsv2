.PHONY: test lint eval

test:
	python -m pytest -q

lint:
	ruff check .

# CPU match eval of the checked-in export. Full 500-turn games.
# Override with AGENT=path GAMES=4 FOUR=4 OUT=eval_results/local_matches.json
AGENT ?= submission/main.py
GAMES ?= 4
FOUR ?= 4
OUT ?= eval_results/local_matches.json

eval:
	python -m orbit_board_bc_train.local_match \
	  --agent $(AGENT) \
	  --games $(GAMES) \
	  --four-player-games $(FOUR) \
	  --out $(OUT)
