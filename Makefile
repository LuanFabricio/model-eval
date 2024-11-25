test:
	mkdir -p logs/models/mobilefacenet
	venv/bin/python main.py models/mobilefacenet.tflite > logs/models/mobilefacenet/_mobilefacenet.log

	mkdir -p logs/models/rafael_student
	venv/bin/python main.py models/rafael_student.tflite > logs/models/rafael_student/_rafael_student.log

	mkdir -p logs/models/triplet_dist_student
	venv/bin/python main.py models/triplet_dist_student.tflite > logs/models/triplet_dist_student/_triplet_dist_student.log

	mkdir -p logs/models/pair_model
	venv/bin/python main.py models/pair_model.tflite > logs/models/pair_model/_pair_model.log

	mkdir -p logs/models/pair_paper_model
	venv/bin/python main.py models/pair_paper_model.tflite > logs/models/pair_paper_model/_pair_paper_model.log
