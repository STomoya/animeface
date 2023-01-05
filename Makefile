
new:
	mkdir animeface/implementations/${name}
	touch animeface/implementations/${name}/model.py
	touch config/config/new.yaml

del:
	rm -rf animeface/implementations/${name}

check:
	docker compose run --rm torch python ${file}

# run
# make run config=gan args="config.train.epochs=10"
run:
	docker compose run --rm torch python -m animeface config=${config} ${args}

# detached run
drun:
	docker compose run --rm -d torch python -m animeface config=${config} ${args}

# resume training from a config file
# make resume config=./path/to/config.yaml
resume:
	docker compose run --rm torch python -m animeface ${config}

# detached resume
dresume:
	docker compose run --rm -d torch python -m animeface ${config}

# run with different docker-compose file
lrun:
	docker compose -f docker-compose.local.yaml run --rm torch python -m animeface ${args}

# detached run with different docker-compose file
dlrun:
	docker compose -f docker-compose.local.yaml run --rm -d torch python -m animeface ${args}
