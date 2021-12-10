
# add folder/files for new implementation
new:
	mkdir implementations/${name}
	mkdir implementations/${name}/result
	touch implementations/${name}/__init__.py
	touch implementations/${name}/model.py
	touch implementations/${name}/utils.py

# reset results folder
reset:
	sudo rm -rf implementations/${name}/result/*

# delete an implementation
del:
	rm -r implementations/${name}

# run a specific single file
check:
	docker-compose run --rm python python ${file}

# RUN

# run
run:
	docker-compose run --rm python python main.py ${ARGS}

# detached run
drun:
	docker-compose run --rm -d python python main.py ${ARGS}

# local run
# use local docker-compose file named `local-dc.yml`
lrun:
	docker-compose -f local-dc.yml run --rm python python main.py ${ARGS}

# local detached run
# use local docker-compose file named `local-dc.yml`
ldrun:
	docker-compose -f local-dc.yml run --rm -d python python main.py ${ARGS}
