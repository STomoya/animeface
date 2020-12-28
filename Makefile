new:
	mkdir implementations/${name}
	mkdir implementations/${name}/result
	touch implementations/${name}/__init__.py
	touch implementations/${name}/model.py
	touch implementations/${name}/utils.py

reset:
	sudo rm -rf implementations/${name}/result/*

del:
	rm -r implementations/${name}

check:
	docker-compose run --rm python python ${file}

run:
	docker-compose run --rm python python main.py