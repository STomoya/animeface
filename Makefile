new:
	mkdir ${name}
	mkdir ${name}/result
	touch ${name}/__init__.py
	touch ${name}/model.py
	touch ${name}/utils.py

del:
	rm -r ${name}

run:
	docker-compose run --rm python python main.py