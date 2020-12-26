new:
	mkdir implementations/${name}
	mkdir implementations/${name}/result
	touch implementations/${name}/__init__.py
	touch implementations/${name}/model.py
	touch implementations/${name}/utils.py

del:
	rm -r implementations/${name}

run:
	docker-compose -f no-wave-image.yml run --rm python python main.py

run_wave:
	docker-compose run --rm python python main.py